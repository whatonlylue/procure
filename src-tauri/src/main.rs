#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! procure desktop shell.
//!
//! Owns the Python sidecar lifecycle: spawn `ProcureHelper serve` on a
//! stable local port with the OS app-data dir, wait for its `PROCURE_URL=`
//! announcement on stdout, open the main window on that URL, and kill the
//! sidecar when the window closes. The UI itself is served by the sidecar,
//! so this crate exposes no JS APIs of its own (the folder picker uses the
//! stock dialog plugin).

use std::sync::Mutex;
use std::time::Duration;

use tauri::{AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

struct SidecarState(Mutex<Option<CommandChild>>);

/// Preferred sidecar port for the desktop window.
///
/// The UI keeps small persistent state in web storage (theme), which is
/// scoped per origin — scheme + host + port. Binding the sidecar to a
/// stable port keeps the window on one origin across restarts so that
/// state survives; an ephemeral port would reset it every launch.
/// Matches `serve --port`'s default and the `devUrl` in tauri.conf.json.
const PREFERRED_SIDECAR_PORT: u16 = 8000;

const LOADING_HTML: &str = "<body style=\"margin:0;background:#141512;color:#989d98;\
  font:14px system-ui;display:flex;align-items:center;justify-content:center;\
  height:100vh\"><span id=\"st\">Starting procure…</span></body>";

/// Update the splash screen's status line (best-effort: silently ignored
/// when the page isn't ready yet; the baked-in text stays meanwhile).
fn set_stage(win: &tauri::WebviewWindow, stage: &str) {
    eprintln!("procure: {stage}");
    let _ = win.eval(format!(
        "var el=document.getElementById('st');if(el)el.textContent={};",
        serde_json::to_string(stage).unwrap_or_default()
    ));
}

/// Pick the port to spawn the sidecar on: the preferred stable port when
/// it is free, otherwise 0 (ephemeral) so a collision can never prevent
/// startup. The probe-then-bind race is closed by a one-time respawn
/// fallback in `boot` (see below).
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

/// Whether a `CloseRequested` event for `label` should quit the app.
///
/// Closing the main window always quits. Closing the splash quits only
/// while the main window doesn't exist yet (the user aborting mid-boot):
/// once main is up, the splash is gone and any late event for it must not
/// take the app down. Pure function so the quit decision is unit-tested
/// (quitting on the splash-to-main transition strands nothing but kills
/// everything: the sidecar is destroyed and the app exits).
fn should_exit_on_close(label: &str, main_exists: bool) -> bool {
    match label {
        "main" => true,
        "splash" => !main_exists,
        _ => false,
    }
}

/// Scan complete lines in `buf` for the sidecar's `PROCURE_URL=` announcement.
/// Consumed lines are drained; returns the URL when found. Pure function so
/// the announce protocol is unit-tested (a missed announcement strands the
/// app on the loading screen).
fn scan_url_line(buf: &mut Vec<u8>) -> Option<String> {
    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
        let line: Vec<u8> = buf.drain(..=pos).collect();
        let text = String::from_utf8_lossy(&line);
        if let Some(url) = text.trim().strip_prefix("PROCURE_URL=") {
            return Some(url.to_string());
        }
    }
    None
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
        if let Some(url) = scan_url_line(&mut buf) {
            return Ok(url);
        }
        if buf.len() > 1_000_000 {
            return Err("sidecar produced 1MB of output without a URL".to_string());
        }
    }
}

fn spawn_sidecar(
    handle: &AppHandle,
    port: u16,
    data_dir: &std::path::Path,
) -> Result<(tokio::sync::mpsc::Receiver<CommandEvent>, CommandChild), String> {
    let port_arg = port.to_string();
    let (rx, child) = handle
        .shell()
        .sidecar("ProcureHelper")
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
    Ok((rx, child))
}

/// Build the main window directly on the sidecar URL.
///
/// This is the long-standing path: the window opens on the live server,
/// with no client-side navigation step in between.
fn build_main_window(handle: &AppHandle, url: tauri::Url) -> Result<(), String> {
    let builder = WebviewWindowBuilder::new(handle, "main", WebviewUrl::External(url))
        // Empty title: macOS renders the window title as OS-level
        // text in the transparent bar, next to the traffic lights.
        .title("")
        .background_color(tauri::window::Color(20, 21, 18, 255))
        .inner_size(1200.0, 800.0)
        .min_inner_size(900.0, 600.0);
    // Overlay (macOS only): no title bar, traffic lights float
    // over the content (which reserves top-left space in CSS).
    // The transparent strip stays natively draggable. Windows
    // keeps its normal title bar. `title_bar_style` does not
    // exist on Windows, so this must stay behind `cfg`.
    #[cfg(target_os = "macos")]
    let builder = builder.title_bar_style(tauri::TitleBarStyle::Overlay);
    builder
        .build()
        .map(|_| ())
        .map_err(|e| format!("create window: {e}"))
}

/// Boot the sidecar and main window without blocking app setup.
///
/// Setup returns immediately so the splash paints at once; the slow
/// onefile unpack + sidecar bind then proceeds in the background
/// (previously setup block_on'd up to 60s with no window, which macOS
/// flags as "not responding"). Once the sidecar announces its URL the
/// splash closes and the main window opens directly on it.
async fn boot(handle: AppHandle) {
    // Splash first: about:blank paints instantly with our text.
    let loading: tauri::Url = "about:blank".parse().expect("about:blank");
    let splash = match WebviewWindowBuilder::new(&handle, "splash", WebviewUrl::External(loading))
        .title("")
        .background_color(tauri::window::Color(20, 21, 18, 255))
        .inner_size(1200.0, 800.0)
        .min_inner_size(900.0, 600.0)
        .build()
    {
        Ok(w) => w,
        Err(e) => {
            eprintln!("procure: create splash: {e}");
            handle.exit(1);
            return;
        }
    };
    let _ = splash.eval(format!(
        "document.open();document.write({});document.close();",
        serde_json::to_string(LOADING_HTML).unwrap_or_default()
    ));
    // Own the sidecar slot from the start so quitting mid-boot (closing
    // the splash) still reaches the child instead of orphaning it.
    handle.manage(SidecarState(Mutex::new(None)));

    let data_dir = match handle.path().app_data_dir() {
        Ok(d) => d,
        Err(e) => {
            set_stage(&splash, &format!("app data dir: {e}"));
            return;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        set_stage(&splash, &format!("create {}: {e}", data_dir.display()));
        return;
    }

    // Preferred stable port first; on failure (a bind race with the probe,
    // or a stale occupant) respawn once on an ephemeral port rather than
    // refusing to start.
    let mut attempt = pick_sidecar_port(PREFERRED_SIDECAR_PORT);
    if attempt == 0 {
        eprintln!(
            "procure: port {PREFERRED_SIDECAR_PORT} in use, \
             using an ephemeral port; UI preferences will not \
             persist this session"
        );
    }
    set_stage(&splash, "Starting server…");
    let store_child = |handle: &AppHandle, child| {
        if let Some(state) = handle.try_state::<SidecarState>() {
            if let Ok(mut guard) = state.0.lock() {
                if let Some(old) = guard.replace(child) {
                    let _ = old.kill();
                }
            }
        }
    };
    let rx = loop {
        let (mut rx0, child0) = match spawn_sidecar(&handle, attempt, &data_dir) {
            Ok(pair) => pair,
            Err(e) if attempt != 0 => {
                eprintln!("procure: sidecar on {attempt} failed ({e}); retrying ephemeral");
                attempt = 0;
                continue;
            }
            Err(e) => {
                set_stage(&splash, &e);
                return;
            }
        };
        store_child(&handle, child0);
        set_stage(&splash, "Waiting for server…");
        match tokio::time::timeout(Duration::from_secs(60), wait_for_url(&mut rx0)).await {
            Ok(Ok(u)) => break (rx0, u),
            _ if attempt != 0 => {
                eprintln!("procure: sidecar on {attempt} never announced; retrying ephemeral");
                attempt = 0;
            }
            Ok(Err(e)) => {
                set_stage(&splash, &e);
                return;
            }
            Err(_) => {
                set_stage(&splash, "timed out waiting for sidecar URL");
                return;
            }
        }
    };
    let (mut rx, url) = rx;
    // Keep draining sidecar output so logging can never block it.
    tauri::async_runtime::spawn(async move {
        while rx.recv().await.is_some() {}
    });

    let parsed: tauri::Url = match url.parse() {
        Ok(u) => u,
        Err(e) => {
            set_stage(&splash, &format!("bad sidecar URL {url:?}: {e}"));
            return;
        }
    };
    // Main first, then tear down the splash. `close()` emits CloseRequested
    // like a user close, which the window-event handler reads as a quit
    // request (killing the sidecar and exiting mid-transition); `destroy()`
    // emits no event. Building main first also means a genuine user close
    // of the splash (pre-main) still quits via the handler below.
    set_stage(&splash, "Opening…");
    if let Err(e) = build_main_window(&handle, parsed) {
        set_stage(&splash, &e);
        return;
    }
    let _ = splash.destroy();
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move { boot(handle).await });
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                let app = window.app_handle().clone();
                let main_exists = app.get_webview_window("main").is_some();
                if should_exit_on_close(window.label(), main_exists) {
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

    #[test]
    fn scan_finds_announcement_among_noise() {
        let mut buf = b"uvicorn running\nPROCURE_URL=http://127.0.0.1:8000\n".to_vec();
        assert_eq!(
            scan_url_line(&mut buf),
            Some("http://127.0.0.1:8000".to_string())
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn scan_waits_for_split_lines() {
        let mut buf = b"PROCURE_URL=http://127.".to_vec();
        assert_eq!(scan_url_line(&mut buf), None);
        // Partial line is kept, not drained.
        assert!(!buf.is_empty());
        buf.extend_from_slice(b"0.0.1:8000\n");
        assert_eq!(
            scan_url_line(&mut buf),
            Some("http://127.0.0.1:8000".to_string())
        );
    }

    #[test]
    fn scan_ignores_lookalike_prefix() {
        let mut buf = b"XPROCURE_URL=http://x/\nPROCURE_URL=http://127.0.0.1:9/\n".to_vec();
        assert_eq!(
            scan_url_line(&mut buf),
            Some("http://127.0.0.1:9/".to_string())
        );
    }

    #[test]
    fn main_close_always_quits() {
        assert!(should_exit_on_close("main", false));
        assert!(should_exit_on_close("main", true));
    }

    #[test]
    fn splash_close_quits_only_before_main_exists() {
        // User aborting mid-boot: quit (and kill the sidecar).
        assert!(should_exit_on_close("splash", false));
        // Splash-to-main transition (or a late splash event once main is
        // up): must NOT quit — this was the startup crash.
        assert!(!should_exit_on_close("splash", true));
    }

    #[test]
    fn unknown_window_close_never_quits() {
        assert!(!should_exit_on_close("about", false));
        assert!(!should_exit_on_close("about", true));
        assert!(!should_exit_on_close("", false));
    }
}
