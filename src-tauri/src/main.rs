#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! procure desktop shell.
//!
//! Owns the Python sidecar lifecycle: spawn `procure-sidecar serve` on a
//! stable local port with the OS app-data dir, wait for its `PROCURE_URL=`
//! announcement on stdout, open the main window on that URL, and kill the
//! sidecar when the window closes. The UI itself is served by the sidecar,
//! so this crate intentionally exposes no JS APIs.

use std::sync::Mutex;
use std::time::Duration;

use tauri::{AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

struct SidecarState(Mutex<Option<CommandChild>>);

/// Preferred sidecar port for the desktop window.
///
/// The UI keeps small persistent state in web storage (greeting rotation,
/// theme), which is scoped per origin — scheme + host + port. Binding the
/// sidecar to a stable port keeps the window on one origin across restarts
/// so that state survives; an ephemeral port would reset it every launch.
/// Matches `serve --port`'s default and the `devUrl` in tauri.conf.json.
const PREFERRED_SIDECAR_PORT: u16 = 8000;

/// Pick the port to spawn the sidecar on: the preferred stable port when
/// it is free, otherwise 0 (ephemeral) so a collision can never prevent
/// startup. The caller announces the ephemeral fallback.
fn pick_sidecar_port(preferred: u16) -> u16 {
    match std::net::TcpListener::bind(("127.0.0.1", preferred)) {
        Ok(probe) => {
            // Release immediately so the sidecar can bind it.
            drop(probe);
            preferred
        }
        Err(_) => 0,
    }
}

fn kill_sidecar(app: &AppHandle) {
    if let Some(state) = app.try_state::<SidecarState>() {
        if let Ok(mut guard) = state.0.lock() {
            if let Some(child) = guard.take() {
                let _ = child.kill();
            }
        }
    }
}

async fn wait_for_url(rx: &mut tokio::sync::mpsc::Receiver<CommandEvent>) -> Result<String, String> {
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let chunk = match rx.recv().await {
            Some(CommandEvent::Stdout(b)) | Some(CommandEvent::Stderr(b)) => b,
            Some(CommandEvent::Terminated(_)) => {
                return Err("sidecar exited before announcing its URL".to_string())
            }
            Some(CommandEvent::Error(e)) => return Err(format!("sidecar error: {e}")),
            Some(_) => continue,
            None => return Err("sidecar output closed before announcing its URL".to_string()),
        };
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let text = String::from_utf8_lossy(&line);
            if let Some(url) = text.trim().strip_prefix("PROCURE_URL=") {
                return Ok(url.to_string());
            }
        }
        if buf.len() > 1_000_000 {
            return Err("sidecar produced 1MB of output without a URL".to_string());
        }
    }
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::block_on(async move {
                let data_dir = handle
                    .path()
                    .app_data_dir()
                    .map_err(|e| format!("app data dir: {e}"))?;
                std::fs::create_dir_all(&data_dir)
                    .map_err(|e| format!("create {}: {e}", data_dir.display()))?;

                let port = pick_sidecar_port(PREFERRED_SIDECAR_PORT);
                if port == 0 {
                    eprintln!(
                        "procure: port {PREFERRED_SIDECAR_PORT} in use, \
                         using an ephemeral port; UI preferences will not \
                         persist this session"
                    );
                }
                let port_arg = port.to_string();
                let (mut rx, child) = handle
                    .shell()
                    .sidecar("procure-sidecar")
                    .map_err(|e| format!("sidecar lookup: {e}"))?
                    .args([
                        "serve",
                        "--port",
                        &port_arg,
                        "--data-dir",
                        &data_dir.to_string_lossy(),
                    ])
                    .spawn()
                    .map_err(|e| format!("sidecar spawn: {e}"))?;
                handle.manage(SidecarState(Mutex::new(Some(child))));

                let url = tokio::time::timeout(Duration::from_secs(60), wait_for_url(&mut rx))
                    .await
                    .map_err(|_| "timed out waiting for sidecar URL".to_string())??;

                // Keep draining sidecar output so logging can never block it.
                tauri::async_runtime::spawn(async move {
                    while rx.recv().await.is_some() {}
                });

                let parsed =
                    url.parse().map_err(|e| format!("bad sidecar URL {url:?}: {e}"))?;
                WebviewWindowBuilder::new(&handle, "main", WebviewUrl::External(parsed))
                    // Empty title: macOS renders the window title as OS-level
                    // text in the transparent bar, next to the traffic lights.
                    .title("")
                    // Overlay: no title bar, traffic lights float over the
                    // content (which reserves top-left space in CSS). The
                    // transparent strip stays natively draggable. Windows
                    // ignores this and keeps its normal title bar.
                    .title_bar_style(tauri::TitleBarStyle::Overlay)
                    .background_color(tauri::window::Color(20, 21, 18, 255))
                    .inner_size(1200.0, 800.0)
                    .min_inner_size(900.0, 600.0)
                    .build()
                    .map_err(|e| format!("create window: {e}"))?;
                Ok::<(), String>(())
            })
            .map_err(|e: String| -> Box<dyn std::error::Error> { e.into() })?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == "main" {
                if let WindowEvent::CloseRequested { .. } = event {
                    let app = window.app_handle().clone();
                    kill_sidecar(&app);
                    app.exit(0);
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("procure failed to start")
        .run(|app, event| {
            if matches!(event, RunEvent::Exit) {
                kill_sidecar(app);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn keeps_preferred_port_when_free() {
        // Discover a free port, release it, and expect it back.
        let free = TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        assert_eq!(pick_sidecar_port(free), free);
    }

    #[test]
    fn falls_back_to_ephemeral_when_preferred_in_use() {
        // Hold a port, then expect the ephemeral fallback (0) for it.
        let held = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = held.local_addr().unwrap().port();
        assert_eq!(pick_sidecar_port(port), 0);
    }
}
