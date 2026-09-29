#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! procure desktop shell.
//!
//! Owns the Python sidecar lifecycle: spawn `procure-sidecar serve` on an
//! ephemeral port with the OS app-data dir, wait for its `PROCURE_URL=`
//! announcement on stdout, open the main window on that URL, and kill the
//! sidecar when the window closes. The UI itself is served by the sidecar,
//! so this crate intentionally exposes no JS APIs.

use std::sync::Mutex;
use std::time::Duration;

use tauri::{AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

struct SidecarState(Mutex<Option<CommandChild>>);

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

                let (mut rx, child) = handle
                    .shell()
                    .sidecar("procure-sidecar")
                    .map_err(|e| format!("sidecar lookup: {e}"))?
                    .args([
                        "serve",
                        "--port",
                        "0",
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
                    .title("procure")
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
