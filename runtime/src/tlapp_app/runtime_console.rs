use super::*;

impl TlAppRuntime {
    pub(super) fn toggle_console(&mut self) {
        self.console.open = !self.console.open;
        self.console.history_cursor = None;
        self.console.edit_target = RuntimeConsoleEditTarget::Command;
        if self.console.open {
            self.keyboard_camera = CameraInputState::default();
            self.mouse_look_held = false;
            self.script_key_f_keyboard = false;
            self.script_key_g_keyboard = false;
            self.cursor_position = None;
            self.camera.set_look_active(&self.window, false);
            self.sync_console_quick_fields_from_runtime();
            self.console_feedback(
                "console opened (Ctrl+F1 toggles, Enter runs command, 'help' lists commands)",
            );
        } else {
            self.console_feedback("console closed");
        }
    }

    pub(super) fn sync_console_quick_fields_from_runtime(&mut self) {
        self.console.quick_fps_cap = self
            .frame_cap_interval
            .map(|itv| format!("{:.0}", 1.0 / itv.as_secs_f32().max(1e-6)))
            .unwrap_or_else(|| "off".to_string());
        self.console.quick_render_distance = self
            .render_distance
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "off".to_string());
        self.console.quick_fsr_sharpness = format!("{:.2}", self.fsr_config.sharpness);
        self.console.quick_msaa = if self.renderer.msaa_sample_count() > 1 {
            format!("{}", self.renderer.msaa_sample_count())
        } else {
            "off".to_string()
        };
    }

    pub(super) fn classify_console_feedback(message: &str) -> RuntimeConsoleLogLevel {
        let lower = message.to_ascii_lowercase();
        if lower.contains("error")
            || lower.contains("failed")
            || lower.contains("invalid")
            || lower.contains("unknown")
            || lower.contains("usage:")
            || lower.contains("out of range")
        {
            RuntimeConsoleLogLevel::Error
        } else {
            RuntimeConsoleLogLevel::Info
        }
    }

    pub(super) fn push_console_log(
        &mut self,
        level: RuntimeConsoleLogLevel,
        message: String,
        print_to_stdout: bool,
    ) {
        self.console.last_feedback = message.clone();
        self.console.log_lines.push(RuntimeConsoleLogLine {
            timestamp: Instant::now(),
            level,
            message: message.clone(),
        });
        if self.console.log_lines.len() > CONSOLE_MAX_LOG_LINES {
            let trim = self.console.log_lines.len() - CONSOLE_MAX_LOG_LINES;
            self.console.log_lines.drain(0..trim);
        }
        let max_scroll = self.max_console_log_scroll();
        self.console.log_scroll = self.console.log_scroll.min(max_scroll);
        if print_to_stdout {
            println!("[tlapp console] {message}");
        }
    }

    pub(super) fn console_feedback(&mut self, message: impl Into<String>) {
        let message = message.into();
        let level = Self::classify_console_feedback(&message);
        self.push_console_log(level, message, true);
    }

    pub(super) fn submit_console_command(&mut self, command: String) -> RuntimeCommand {
        let command = command.trim().to_string();
        if !command.is_empty() {
            self.console.history.push(command.clone());
            if self.console.history.len() > 128 {
                let trim = self.console.history.len() - 128;
                self.console.history.drain(0..trim);
            }
            self.push_console_log(RuntimeConsoleLogLevel::Info, format!("> {command}"), false);
        }
        self.console.input_line.clear();
        self.console.history_cursor = None;
        self.console.log_scroll = 0;
        if command.is_empty() {
            return RuntimeCommand::Consumed;
        }
        self.execute_console_command(&command)
    }

    pub(super) fn selected_console_edit_buffer_mut(&mut self) -> &mut String {
        match self.console.edit_target {
            RuntimeConsoleEditTarget::Command => &mut self.console.input_line,
            RuntimeConsoleEditTarget::FpsCap => &mut self.console.quick_fps_cap,
            RuntimeConsoleEditTarget::RenderDistance => &mut self.console.quick_render_distance,
            RuntimeConsoleEditTarget::FsrSharpness => &mut self.console.quick_fsr_sharpness,
            RuntimeConsoleEditTarget::Msaa => &mut self.console.quick_msaa,
        }
    }

    pub(super) fn apply_active_console_edit_target(&mut self) -> RuntimeCommand {
        match self.console.edit_target {
            RuntimeConsoleEditTarget::Command => {
                self.submit_console_command(self.console.input_line.clone())
            }
            RuntimeConsoleEditTarget::FpsCap => {
                let value = self.console.quick_fps_cap.trim();
                match parse_fps_cap(value) {
                    Ok(cap) => {
                        self.apply_fps_cap_runtime(cap);
                        match cap {
                            Some(fps) => self.console_feedback(format!("fps cap set to {fps:.1}")),
                            None => self.console_feedback("fps cap disabled"),
                        }
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
                RuntimeCommand::Consumed
            }
            RuntimeConsoleEditTarget::RenderDistance => {
                let value = self.console.quick_render_distance.trim();
                match parse_render_distance(value) {
                    Ok(distance) => {
                        self.render_distance = distance;
                        if let Some(value) = distance {
                            self.render_distance_min = (value * 0.72).clamp(28.0, value);
                            self.render_distance_max =
                                (value * 1.55).clamp(value, value.max(220.0));
                            self.console_feedback(format!("render distance set to {value:.1}"));
                        } else {
                            self.console_feedback("render distance disabled");
                        }
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
                RuntimeCommand::Consumed
            }
            RuntimeConsoleEditTarget::FsrSharpness => {
                let value = self.console.quick_fsr_sharpness.trim();
                match parse_fsr_sharpness(value) {
                    Ok(sharpness) => {
                        self.fsr_config.sharpness = sharpness;
                        self.renderer.set_fsr_config(&self.queue, self.fsr_config);
                        self.console_feedback(format!("fsr sharpness set to {sharpness:.2}"));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
                RuntimeCommand::Consumed
            }
            RuntimeConsoleEditTarget::Msaa => {
                let value = self.console.quick_msaa.trim();
                match parse_msaa(value) {
                    Ok(count) => {
                        self.renderer.set_msaa_sample_count(&self.device, count);
                        let label = if count > 1 {
                            format!("msaa {count}x")
                        } else {
                            "msaa off".to_string()
                        };
                        self.console_feedback(format!("{label} applied"));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
                RuntimeCommand::Consumed
            }
        }
    }

    pub(super) fn handle_console_window_event(&mut self, event: &WindowEvent) -> bool {
        if !self.console.open {
            return false;
        }

        match event {
            WindowEvent::MouseWheel { delta, .. } => {
                let (delta_y, line_steps) = match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        let steps = y.abs().round().max(1.0) as i32;
                        (*y as f64, steps)
                    }
                    MouseScrollDelta::PixelDelta(pixels) => {
                        let steps = (pixels.y.abs() / CONSOLE_WHEEL_PIXELS_PER_LINE)
                            .round()
                            .max(1.0) as i32;
                        (pixels.y, steps)
                    }
                };
                if delta_y > 0.0 {
                    self.scroll_console_logs(line_steps);
                } else if delta_y < 0.0 {
                    self.scroll_console_logs(-line_steps);
                }
                true
            }
            _ => false,
        }
    }

    #[inline]
    pub(super) fn max_console_log_scroll(&self) -> usize {
        self.visible_console_log_count().saturating_sub(1)
    }

    pub(super) fn scroll_console_logs(&mut self, delta_lines: i32) {
        if delta_lines == 0 {
            return;
        }
        let max_scroll = self.max_console_log_scroll();
        if delta_lines > 0 {
            self.console.log_scroll = self
                .console
                .log_scroll
                .saturating_add(delta_lines as usize)
                .min(max_scroll);
        } else {
            self.console.log_scroll = self
                .console
                .log_scroll
                .saturating_sub(delta_lines.unsigned_abs() as usize);
        }
    }

    pub(super) fn visible_console_log_count(&self) -> usize {
        let mut count = self
            .console
            .log_lines
            .iter()
            .filter(|line| self.console.log_filter.matches(line.level))
            .count();
        if let Some(limit) = self.console.log_tail_limit {
            count = count.min(limit);
        }
        count
    }

    pub(super) fn resolve_console_candidate_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        let trimmed = raw_path.trim();
        if trimmed.is_empty() {
            return Err("path must not be empty".to_string());
        }
        let candidate = PathBuf::from(trimmed);
        Ok(if candidate.is_absolute() {
            candidate
        } else {
            self.file_io_root.join(candidate)
        })
    }

    pub(super) fn resolve_console_existing_path(
        &self,
        raw_path: &str,
        op: &str,
    ) -> Result<PathBuf, String> {
        let candidate = self.resolve_console_candidate_path(raw_path)?;
        let canonical = candidate
            .canonicalize()
            .map_err(|err| format!("{op}: failed to access '{raw_path}': {err}"))?;
        if !canonical.starts_with(&self.file_io_root) {
            return Err(format!(
                "{op}: denied (path escapes workspace root '{}')",
                self.file_io_root.display()
            ));
        }
        Ok(canonical)
    }

    pub(super) fn emit_console_text_lines(&mut self, prefix: &str, text: &str, max_lines: usize) {
        let mut lines = text.lines();
        let mut emitted = 0usize;
        while emitted < max_lines {
            let Some(line) = lines.next() else {
                break;
            };
            self.console_feedback(format!("{prefix}{line}"));
            emitted += 1;
        }
        if emitted == 0 {
            self.console_feedback(format!("{prefix}<empty>"));
        } else if lines.next().is_some() {
            self.console_feedback(format!("{prefix}... output truncated to {max_lines} lines"));
        }
    }

    pub(super) fn read_file_prefix_bytes(
        &self,
        path: &Path,
        max_bytes: usize,
    ) -> Result<(Vec<u8>, bool), String> {
        let mut file = fs::File::open(path)
            .map_err(|err| format!("failed to open '{}': {err}", path.display()))?;
        let file_len = file.metadata().ok().map(|meta| meta.len());
        let mut buf = vec![0u8; max_bytes];
        let read_len = file
            .read(&mut buf)
            .map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
        buf.truncate(read_len);
        let truncated = file_len.map(|len| len > read_len as u64).unwrap_or(false);
        Ok((buf, truncated))
    }

    pub(super) fn read_file_tail_window(
        &self,
        path: &Path,
        max_window_bytes: usize,
    ) -> Result<(Vec<u8>, bool), String> {
        let mut file = fs::File::open(path)
            .map_err(|err| format!("failed to open '{}': {err}", path.display()))?;
        let file_len = file
            .metadata()
            .map(|meta| meta.len())
            .map_err(|err| format!("failed to query metadata '{}': {err}", path.display()))?;
        if file_len == 0 {
            return Ok((Vec::new(), false));
        }
        let start_offset = file_len.saturating_sub(max_window_bytes as u64);
        file.seek(SeekFrom::Start(start_offset))
            .map_err(|err| format!("failed to seek '{}': {err}", path.display()))?;
        let mut buf = Vec::with_capacity((file_len - start_offset) as usize);
        file.read_to_end(&mut buf)
            .map_err(|err| format!("failed to read tail '{}': {err}", path.display()))?;
        Ok((buf, start_offset > 0))
    }

    pub(super) fn run_file_find(
        &mut self,
        path: &Path,
        pattern: &str,
        max_matches: usize,
        max_bytes: usize,
        case_insensitive: bool,
    ) {
        match self.read_file_prefix_bytes(path, max_bytes) {
            Ok((bytes, truncated)) => {
                let text = String::from_utf8_lossy(&bytes);
                let mut total_matches = 0usize;
                let mut shown = 0usize;
                let pattern_cmp = if case_insensitive {
                    pattern.to_ascii_lowercase()
                } else {
                    String::new()
                };
                self.console_feedback(format!(
                    "{} '{}' pattern='{}' scan={} byte(s) max_matches={}{}",
                    if case_insensitive {
                        "file.findi"
                    } else {
                        "file.find"
                    },
                    path.display(),
                    pattern,
                    bytes.len(),
                    max_matches,
                    if truncated {
                        ", truncated scan window"
                    } else {
                        ""
                    }
                ));
                for (line_idx, line) in text.lines().enumerate() {
                    let matched = if case_insensitive {
                        line.to_ascii_lowercase().contains(&pattern_cmp)
                    } else {
                        line.contains(pattern)
                    };
                    if !matched {
                        continue;
                    }
                    total_matches += 1;
                    if shown < max_matches {
                        shown += 1;
                        self.console_feedback(format!("  L{}: {}", line_idx + 1, line));
                    }
                }
                if total_matches == 0 {
                    self.console_feedback("  <no matches>");
                } else if total_matches > max_matches {
                    self.console_feedback(format!(
                        "  ... {} additional match(es) hidden",
                        total_matches - max_matches
                    ));
                }
            }
            Err(err) => self.console_feedback(format!(
                "{}: {err}",
                if case_insensitive {
                    "file.findi"
                } else {
                    "file.find"
                }
            )),
        }
    }

    pub(super) fn run_file_find_regex(
        &mut self,
        path: &Path,
        pattern: &str,
        max_matches: usize,
        max_bytes: usize,
    ) {
        let regex = match RegexBuilder::new(pattern).size_limit(1_000_000).build() {
            Ok(regex) => regex,
            Err(err) => {
                self.console_feedback(format!("file.findr: invalid regex '{pattern}': {err}"));
                return;
            }
        };
        match self.read_file_prefix_bytes(path, max_bytes) {
            Ok((bytes, truncated)) => {
                let text = String::from_utf8_lossy(&bytes);
                let mut total_matches = 0usize;
                let mut shown = 0usize;
                self.console_feedback(format!(
                    "file.findr '{}' regex='{}' scan={} byte(s) max_matches={}{}",
                    path.display(),
                    pattern,
                    bytes.len(),
                    max_matches,
                    if truncated {
                        ", truncated scan window"
                    } else {
                        ""
                    }
                ));
                for (line_idx, line) in text.lines().enumerate() {
                    if !regex.is_match(line) {
                        continue;
                    }
                    total_matches += 1;
                    if shown < max_matches {
                        shown += 1;
                        self.console_feedback(format!("  L{}: {}", line_idx + 1, line));
                    }
                }
                if total_matches == 0 {
                    self.console_feedback("  <no matches>");
                } else if total_matches > max_matches {
                    self.console_feedback(format!(
                        "  ... {} additional match(es) hidden",
                        total_matches - max_matches
                    ));
                }
            }
            Err(err) => self.console_feedback(format!("file.findr: {err}")),
        }
    }

    pub(super) fn run_file_grep(
        &mut self,
        path: &Path,
        pattern: &str,
        context: usize,
        max_matches: usize,
        max_bytes: usize,
    ) {
        match self.read_file_prefix_bytes(path, max_bytes) {
            Ok((bytes, truncated)) => {
                let text = String::from_utf8_lossy(&bytes);
                let lines = text.lines().collect::<Vec<_>>();
                let mut total_matches = 0usize;
                let mut shown_matches = 0usize;
                let mut printed = vec![false; lines.len()];
                self.console_feedback(format!(
                    "file.grep '{}' pattern='{}' context={} scan={} byte(s) max_matches={}{}",
                    path.display(),
                    pattern,
                    context,
                    bytes.len(),
                    max_matches,
                    if truncated {
                        ", truncated scan window"
                    } else {
                        ""
                    }
                ));
                for (idx, line) in lines.iter().enumerate() {
                    if !line.contains(pattern) {
                        continue;
                    }
                    total_matches += 1;
                    if shown_matches >= max_matches {
                        continue;
                    }
                    shown_matches += 1;
                    let start = idx.saturating_sub(context);
                    let end = (idx + context).min(lines.len().saturating_sub(1));
                    for i in start..=end {
                        if printed[i] {
                            continue;
                        }
                        printed[i] = true;
                        let marker = if i == idx { ">" } else { " " };
                        self.console_feedback(format!(" {marker} L{}: {}", i + 1, lines[i]));
                    }
                }
                if total_matches == 0 {
                    self.console_feedback("  <no matches>");
                } else if total_matches > max_matches {
                    self.console_feedback(format!(
                        "  ... {} additional match(es) hidden",
                        total_matches - max_matches
                    ));
                }
            }
            Err(err) => self.console_feedback(format!("file.grep: {err}")),
        }
    }

    pub(super) fn poll_console_tail_follow(&mut self) {
        let (path, cursor, poll_interval, max_lines_per_poll, partial_line, last_poll, last_error) =
            match self.console.tail_follow.as_ref() {
                Some(follow) => (
                    follow.path.clone(),
                    follow.cursor,
                    follow.poll_interval,
                    follow.max_lines_per_poll,
                    follow.partial_line.clone(),
                    follow.last_poll,
                    follow.last_error.clone(),
                ),
                None => return,
            };

        let now = Instant::now();
        if now.saturating_duration_since(last_poll) < poll_interval {
            return;
        }

        let mut next_cursor = cursor;
        let mut next_partial = partial_line;
        let mut next_error: Option<String> = None;

        let metadata = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(err) => {
                let msg = format!("tailf metadata failed for '{}': {err}", path.display());
                if last_error.as_deref() != Some(msg.as_str()) {
                    self.console_feedback(format!("[tailf] {msg}"));
                }
                if let Some(follow) = self.console.tail_follow.as_mut() {
                    follow.last_poll = now;
                    follow.last_error = Some(msg);
                }
                return;
            }
        };

        if !metadata.is_file() {
            let msg = format!("tailf target is no longer a file: '{}'", path.display());
            if last_error.as_deref() != Some(msg.as_str()) {
                self.console_feedback(format!("[tailf] {msg}"));
            }
            if let Some(follow) = self.console.tail_follow.as_mut() {
                follow.last_poll = now;
                follow.last_error = Some(msg);
            }
            return;
        }

        let file_len = metadata.len();
        if file_len < next_cursor {
            next_cursor = 0;
            next_partial.clear();
            self.console_feedback(format!(
                "[tailf] file was truncated/rotated, rewinding '{}'",
                path.display()
            ));
        }

        let bytes_to_read_u64 =
            (file_len - next_cursor).min(CONSOLE_FILE_TAIL_MAX_WINDOW_BYTES as u64);
        if bytes_to_read_u64 > 0 {
            match fs::File::open(&path) {
                Ok(mut file) => {
                    if let Err(err) = file.seek(SeekFrom::Start(next_cursor)) {
                        let msg = format!("tailf seek failed '{}': {err}", path.display());
                        if last_error.as_deref() != Some(msg.as_str()) {
                            self.console_feedback(format!("[tailf] {msg}"));
                        }
                        next_error = Some(msg);
                    } else {
                        let mut buf = vec![0u8; bytes_to_read_u64 as usize];
                        match file.read_exact(&mut buf) {
                            Ok(()) => {
                                next_cursor = next_cursor.saturating_add(bytes_to_read_u64);
                                let chunk = String::from_utf8_lossy(&buf);
                                next_partial.push_str(&chunk);

                                let mut lines = Vec::new();
                                while let Some(pos) = next_partial.find('\n') {
                                    let mut line = next_partial[..pos].to_string();
                                    if line.ends_with('\r') {
                                        line.pop();
                                    }
                                    lines.push(line);
                                    next_partial.drain(..=pos);
                                }
                                if lines.len() > max_lines_per_poll {
                                    let skipped = lines.len() - max_lines_per_poll;
                                    self.console_feedback(format!(
                                        "[tailf] ... {skipped} line(s) skipped this poll"
                                    ));
                                }
                                let start = lines.len().saturating_sub(max_lines_per_poll);
                                for line in lines.into_iter().skip(start) {
                                    self.console_feedback(format!("[tailf] {line}"));
                                }
                                if next_partial.len() > 8 * 1024 {
                                    let keep_from = next_partial.len().saturating_sub(8 * 1024);
                                    next_partial = next_partial[keep_from..].to_string();
                                }
                            }
                            Err(err) => {
                                let msg = format!("tailf read failed '{}': {err}", path.display());
                                if last_error.as_deref() != Some(msg.as_str()) {
                                    self.console_feedback(format!("[tailf] {msg}"));
                                }
                                next_error = Some(msg);
                            }
                        }
                    }
                }
                Err(err) => {
                    let msg = format!("tailf open failed '{}': {err}", path.display());
                    if last_error.as_deref() != Some(msg.as_str()) {
                        self.console_feedback(format!("[tailf] {msg}"));
                    }
                    next_error = Some(msg);
                }
            }
        }

        if let Some(follow) = self.console.tail_follow.as_mut() {
            follow.cursor = next_cursor;
            follow.partial_line = next_partial;
            follow.last_poll = now;
            follow.last_error = next_error;
        }
    }

    pub(super) fn poll_console_file_watch(&mut self) {
        let (path, poll_interval, last_modified, last_len, last_poll, last_error) =
            match self.console.file_watch.as_ref() {
                Some(watch) => (
                    watch.path.clone(),
                    watch.poll_interval,
                    watch.last_modified,
                    watch.last_len,
                    watch.last_poll,
                    watch.last_error.clone(),
                ),
                None => return,
            };

        let now = Instant::now();
        if now.saturating_duration_since(last_poll) < poll_interval {
            return;
        }

        let metadata = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(err) => {
                let msg = format!("watch metadata failed for '{}': {err}", path.display());
                if last_error.as_deref() != Some(msg.as_str()) {
                    self.console_feedback(format!("[watch] {msg}"));
                }
                if let Some(watch) = self.console.file_watch.as_mut() {
                    watch.last_poll = now;
                    watch.last_error = Some(msg);
                }
                return;
            }
        };

        if !metadata.is_file() {
            let msg = format!("watch target is no longer a file: '{}'", path.display());
            if last_error.as_deref() != Some(msg.as_str()) {
                self.console_feedback(format!("[watch] {msg}"));
            }
            if let Some(watch) = self.console.file_watch.as_mut() {
                watch.last_poll = now;
                watch.last_error = Some(msg);
            }
            return;
        }

        let modified = metadata.modified().ok();
        let len = metadata.len();
        let changed = len != last_len || modified != last_modified;
        if changed {
            self.console_feedback(format!(
                "[watch] changed '{}' size={} (delta={:+})",
                path.display(),
                len,
                len as i64 - last_len as i64
            ));
        }

        if let Some(watch) = self.console.file_watch.as_mut() {
            watch.last_modified = modified;
            watch.last_len = len;
            watch.last_poll = now;
            watch.last_error = None;
        }
    }

    pub(super) fn console_status_line(&self) -> String {
        let perf_contract = self.performance_contract_evaluation().compact_summary();
        let fsr = self.renderer.fsr_status();
        let rt = self.renderer.ray_tracing_status();
        let fps_cap = self
            .frame_cap_interval
            .map(|itv| format!("{:.0}", 1.0 / itv.as_secs_f32().max(1e-6)))
            .unwrap_or_else(|| "off".to_string());
        let render_distance = self
            .render_distance
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "off".to_string());
        let log_tail = self
            .console
            .log_tail_limit
            .map(|n| n.to_string())
            .unwrap_or_else(|| "off".to_string());
        let log_filter = match self.console.log_filter {
            RuntimeConsoleLogFilter::All => "all",
            RuntimeConsoleLogFilter::Info => "info",
            RuntimeConsoleLogFilter::Error => "error",
        };
        let substeps = self
            .manual_max_substeps
            .map(|n| format!("manual:{n}"))
            .unwrap_or_else(|| format!("auto:{}", self.max_substeps));
        let tailf_state = self
            .console
            .tail_follow
            .as_ref()
            .map(|f| f.path.display().to_string())
            .unwrap_or_else(|| "off".to_string());
        let watch_state = self
            .console
            .file_watch
            .as_ref()
            .map(|w| w.path.display().to_string())
            .unwrap_or_else(|| "off".to_string());
        format!(
            "scene_mode={} tile[chunks={} dirty={} vis={} draw={} cull={}] pipeline={} bridge_path={} queued_plan_depth={} bridge_pump_published={} bridge_pump_drained={} physics_lag_frames={} bridge_fallback={} fps_cap={fps_cap} vsync={:?} rt={:?}/{} fsr={:?}/{} scale={:.2} sharpness={:.2} render_distance={} adaptive_distance={:?} distance_blur={:?} sim_paused={} step_budget={} substeps={} log_filter={} log_tail={} tailf={} watch={} script_vars={} script_calls={} {}",
            self.scene_mode().as_str(),
            self.tile_world_frame.loaded_chunks,
            self.tile_world_frame.dirty_chunks,
            self.tile_world_frame.visible_chunks,
            self.tile_world_frame.emitted_tiles,
            self.tile_world_frame.culled_tiles,
            self.pipeline_mode.as_str(),
            self.runtime_bridge_telemetry.bridge_path.as_str(),
            self.runtime_bridge_telemetry.queued_plan_depth,
            self.runtime_bridge_telemetry.bridge_pump_published,
            self.runtime_bridge_telemetry.bridge_pump_drained,
            self.runtime_bridge_telemetry.physics_lag_frames,
            self.runtime_bridge_telemetry.used_fallback_plan,
            self.present_mode,
            self.rt_mode,
            if rt.active { "on" } else { "off" },
            fsr.requested_mode,
            if fsr.active { "on" } else { "off" },
            fsr.render_scale,
            fsr.sharpness,
            render_distance,
            self.adaptive_distance_enabled,
            self.distance_blur_mode,
            self.simulation_paused,
            self.simulation_step_budget,
            substeps,
            log_filter,
            log_tail,
            tailf_state,
            watch_state,
            self.console.script_vars.len(),
            self.console.script_statements.len(),
            perf_contract,
        )
    }

    pub(super) fn performance_contract_evaluation(&self) -> PerformanceContractEvaluation {
        evaluate_performance_contract(
            self.scene.live_ball_count(),
            self.fps_tracker.snapshot(),
            self.tick_hz,
            self.frame_time_jitter_ema_ms,
            self.runtime_bridge_telemetry.physics_lag_frames,
        )
    }

    pub(super) fn performance_contract_evaluation_for(
        &self,
        scenario: Option<PerformanceContractScenario>,
    ) -> PerformanceContractEvaluation {
        let live_balls = self.scene.live_ball_count();
        let fps = self.fps_tracker.snapshot();
        let tick_hz = self.tick_hz;
        let jitter_ms = self.frame_time_jitter_ema_ms;
        let lag_frames = self.runtime_bridge_telemetry.physics_lag_frames;
        if let Some(scenario) = scenario {
            evaluate_performance_contract_for_scenario(
                scenario, live_balls, fps, tick_hz, jitter_ms, lag_frames,
            )
        } else {
            evaluate_performance_contract(live_balls, fps, tick_hz, jitter_ms, lag_frames)
        }
    }

    pub(super) fn console_title_suffix(&self) -> String {
        if !self.console.open {
            return String::new();
        }
        if self.console.input_line.is_empty() {
            return " | CLI ready".to_string();
        }
        let mut preview = self.console.input_line.clone();
        if preview.chars().count() > 42 {
            preview = preview.chars().take(42).collect::<String>() + "...";
        }
        format!(" | CLI> {preview}")
    }

    pub(super) fn expand_console_script_vars(&self, statement: &str) -> Result<String, String> {
        let mut out = String::with_capacity(statement.len() + 16);
        let chars = statement.chars().collect::<Vec<_>>();
        let mut i = 0usize;
        while i < chars.len() {
            let ch = chars[i];
            if ch != '$' {
                out.push(ch);
                i += 1;
                continue;
            }
            let start = i + 1;
            let mut end = start;
            while end < chars.len() && (chars[end].is_ascii_alphanumeric() || chars[end] == '_') {
                end += 1;
            }
            if end == start {
                out.push(ch);
                i += 1;
                continue;
            }
            let name = chars[start..end].iter().collect::<String>();
            let Some(value) = self.console.script_vars.get(&name) else {
                return Err(format!("unknown script var '${name}'"));
            };
            out.push_str(value);
            i = end;
        }
        Ok(out)
    }

    pub(super) fn evaluate_console_statement(
        &self,
        statement: &str,
        tile_lookup: Option<&dyn TlscriptTileLookup>,
    ) -> Result<TlscriptShowcaseFrameOutput, String> {
        let source = format!("@export\ndef showcase_tick():\n    {statement}\n");
        let compile = compile_tlscript_showcase(&source, TlscriptShowcaseConfig::default());
        if !compile.errors.is_empty() {
            return Err(compile.errors.join(" | "));
        }
        let Some(program) = compile.program else {
            return Err("script statement produced no runnable program".to_string());
        };
        let contact_snapshot = {
            let world = self.world.borrow();
            let broadphase = world.broadphase().stats();
            let narrowphase = world.narrowphase().stats();
            TlscriptShowcaseContactSnapshot {
                contact_pairs: broadphase.candidate_pairs,
                contact_manifolds: narrowphase.manifolds,
            }
        };
        let mut output = if let Some(tile_lookup) = tile_lookup {
            program.evaluate_frame_with_controls_and_tile_lookup_and_contacts(
                TlscriptShowcaseFrameInput {
                    frame_index: self.script_frame_index,
                    live_balls: self.scene.live_ball_count(),
                    spawned_this_tick: self.script_last_spawned,
                    key_f_down: self.script_key_f_keyboard || self.gamepad.action_f_down(),
                },
                TlscriptShowcaseControlInput::default(),
                Some(tile_lookup),
                contact_snapshot,
            )
        } else {
            let runtime_tile_lookup =
                |x: i32, y: i32| self.tile_world_2d.tile(TileCoord2d::new(x, y));
            program.evaluate_frame_with_controls_and_tile_lookup_and_contacts(
                TlscriptShowcaseFrameInput {
                    frame_index: self.script_frame_index,
                    live_balls: self.scene.live_ball_count(),
                    spawned_this_tick: self.script_last_spawned,
                    key_f_down: self.script_key_f_keyboard || self.gamepad.action_f_down(),
                },
                TlscriptShowcaseControlInput::default(),
                Some(&runtime_tile_lookup),
                contact_snapshot,
            )
        };
        if !compile.warnings.is_empty() {
            output.warnings.extend(
                compile
                    .warnings
                    .into_iter()
                    .map(|warning| format!("compile warning: {warning}")),
            );
        }
        Ok(output)
    }

    pub(super) fn sanitize_console_overlay_output(output: &mut TlscriptShowcaseFrameOutput) {
        output.performance_preset = None;
        output.gfx_profile = None;
        output.camera_translate_delta = None;
        output.camera_rotate_delta_deg = None;
        output.camera_move_axis = None;
        output.camera_look_delta = None;
        output.camera_sprint = None;
        output.camera_look_active = None;
        output.camera_reset_pose = false;
        output.dispatch_decision = None;
        output.aborted_early = false;
    }

    pub(super) fn normalize_script_call_statement(raw: &str) -> Result<String, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("script.call requires a function call".to_string());
        }
        if trimmed.contains('(') {
            return Ok(trimmed.trim_end_matches(';').to_string());
        }

        let mut tokens = trimmed.split_whitespace();
        let Some(name) = tokens.next() else {
            return Err("script.call requires a function call".to_string());
        };
        if !is_valid_tlscript_ident(name) {
            return Err(format!("invalid function name '{name}'"));
        }
        let args = tokens.collect::<Vec<_>>();
        if args.is_empty() {
            return Ok(format!("{name}()"));
        }
        Ok(format!("{name}({})", args.join(", ")))
    }

    pub(super) fn rebuild_console_script_overlay(&mut self) -> Result<Vec<String>, String> {
        let mut rebuilt = empty_showcase_output();
        let runtime_tile_lookup = |x: i32, y: i32| self.tile_world_2d.tile(TileCoord2d::new(x, y));
        let mut notes = Vec::new();
        for (index, statement_template) in self.console.script_statements.iter().enumerate() {
            let expanded = self
                .expand_console_script_vars(statement_template)
                .map_err(|err| {
                    format!("statement[{index}] '{statement_template}' variable error: {err}")
                })?;
            let overlay_lookup = TlscriptOverlayTileLookup::new(
                Some(&runtime_tile_lookup),
                &rebuilt.tile_mutations,
                &rebuilt.tile_fills,
            );
            let mut output = self
                .evaluate_console_statement(&expanded, Some(&overlay_lookup))
                .map_err(|err| format!("statement[{index}] '{expanded}' failed: {err}"))?;
            if !output.warnings.is_empty() {
                notes.push(format!(
                    "statement[{index}] warnings: {}",
                    output.warnings.join(" | ")
                ));
            }
            Self::sanitize_console_overlay_output(&mut output);
            merge_showcase_output(&mut rebuilt, output, CONSOLE_SCRIPT_INDEX);
        }
        rebuilt.warnings.clear();
        self.console.script_overlay = rebuilt;
        Ok(notes)
    }

    pub(super) fn execute_console_command(&mut self, command: &str) -> RuntimeCommand {
        let trimmed = command.trim();
        if trimmed.is_empty() {
            self.console_feedback("empty command");
            return RuntimeCommand::Consumed;
        }

        let mut parts = trimmed.split_whitespace();
        let Some(head) = parts.next() else {
            self.console_feedback("empty command");
            return RuntimeCommand::Consumed;
        };
        let head_lc = head.to_ascii_lowercase();
        match head_lc.as_str() {
            "help" | "?" => {
                if let Some(topic) = parts.next() {
                    let topic = topic.to_ascii_lowercase();
                    let topic_list = match topic.as_str() {
                        "file" => Some(vec![
                            "file.exists <path>",
                            "file.head <path> [lines]",
                            "file.tail <path> [lines] [max_bytes]",
                            "file.tailf <path>|stop [poll_ms] [max_lines]",
                            "file.watch <path>|stop [poll_ms]",
                            "file.read <path> [max_bytes]",
                            "file.list <dir> [limit]",
                            "file.find <path> <pattern> [max_matches] [max_bytes]",
                            "file.findi <path> <pattern> [max_matches] [max_bytes]",
                            "file.findr <path> <regex> [max_matches] [max_bytes]",
                            "file.grep <path> <pattern> [context] [max_matches] [max_bytes]",
                        ]),
                        "gfx" => Some(vec![
                            "gfx.status",
                            "gfx.vsync <auto|on|off>",
                            "gfx.fps_cap <off|N>",
                            "gfx.rt <off|auto|on>",
                            "gfx.fsr <off|auto|on>",
                            "gfx.fsr_quality <native|ultra|quality|balanced|performance>",
                            "gfx.fsr_sharpness <0..1>",
                            "gfx.fsr_scale <auto|0.5..1>",
                            "gfx.msaa <off|2|4>",
                            "gfx.profile <low|med|high|ultra>",
                            "gfx.render_distance <off|N>",
                            "gfx.adaptive_distance <auto|on|off>",
                            "gfx.distance_blur <auto|on|off>",
                        ]),
                        "sim" => Some(vec![
                            "sim.status",
                            "sim.pause",
                            "sim.resume",
                            "sim.step <n>",
                            "sim.reset",
                            "scene.mode <3d|2d>",
                            "tile.status",
                            "tile.set <x y id>",
                            "tile.dig <x y>",
                            "tile.fill <x0 y0 x1 y1 id>",
                            "scene.reload | script.reload | sprite.reload",
                            "perf.snapshot",
                            "perf.contract [8k|30k|60k]",
                            "perf.report [8k|30k|60k]",
                            "perf.preset <8k|30k|60k>",
                            "phys.gravity <x y z>",
                            "phys.substeps <auto|n>",
                        ]),
                        "script" => Some(vec![
                            "script.var <name> <expr>",
                            "script.unset <name>",
                            "script.vars",
                            "script.call <fn(args)>",
                            "script.exec <stmt>",
                            "script.uncall <idx|all>",
                            "script.list",
                            "script.clear",
                        ]),
                        "cam" | "camera" => {
                            Some(vec!["cam.speed <v>", "cam.sens <v>", "cam.reset"])
                        }
                        "log" => Some(vec![
                            "log.clear",
                            "log.tail <off|n>",
                            "log.level <all|info|error>",
                        ]),
                        _ => None,
                    };
                    if let Some(commands) = topic_list {
                        self.console_feedback(format!("help {topic}:"));
                        for cmd in commands {
                            self.console_feedback(format!("  {cmd}"));
                        }
                    } else {
                        self.console_feedback(
                            "unknown help topic (use: file|gfx|sim|script|cam|log)",
                        );
                    }
                } else {
                    self.console_feedback(format!(
                        "commands: {}",
                        CONSOLE_HELP_COMMANDS.join(" | ")
                    ));
                }
            }
            "version" | "ver" => {
                let query = parts.next().unwrap_or("all");
                if query.eq_ignore_ascii_case("all") {
                    self.console_feedback(format!("{ENGINE_ID} engine v{ENGINE_VERSION}"));
                    for entry in tileline_version_entries() {
                        self.console_feedback(format!("  {:<10} v{}", entry.module, entry.version));
                    }
                } else if let Some(entry) = resolve_tileline_version_query(query) {
                    self.console_feedback(format!("{} v{}", entry.module, entry.version));
                } else {
                    self.console_feedback(
                        "usage: version [module|all] (module: tileline|runtime|tl-core|mps|gms|mgs|nps|paradoxpe)",
                    );
                }
            }
            "status" | "gfx.status" => {
                self.console_feedback(self.console_status_line());
            }
            "sim.status" => {
                self.console_feedback(format!(
                    "sim mode={} paused={} step_budget={} tick_hz={:.1} max_substeps={} manual_substeps={:?}",
                    self.scene_mode().as_str(),
                    self.simulation_paused,
                    self.simulation_step_budget,
                    self.tick_hz,
                    self.max_substeps,
                    self.manual_max_substeps
                ));
            }
            "sim.pause" => {
                self.simulation_paused = true;
                self.console_feedback("simulation paused");
            }
            "sim.resume" => {
                self.simulation_paused = false;
                self.simulation_step_budget = 0;
                self.console_feedback("simulation resumed");
            }
            "sim.step" => {
                let steps = match parts.next() {
                    Some(value) => match parse_console_u32_in_range(value, "sim.step", 1, 240) {
                        Ok(parsed) => parsed,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => 1,
                };
                self.simulation_paused = true;
                self.simulation_step_budget = self.simulation_step_budget.saturating_add(steps);
                self.console_feedback(format!(
                    "simulation step budget increased by {steps} (pending={})",
                    self.simulation_step_budget
                ));
            }
            "sim.reset" => {
                self.reset_simulation_state();
                self.console_feedback("simulation reset (world + scene)");
            }
            "scene.mode" => {
                let Some(raw_mode) = parts.next() else {
                    self.console_feedback("usage: scene.mode <3d|2d>");
                    return RuntimeCommand::Consumed;
                };
                match RuntimeSceneMode::from_str(raw_mode) {
                    Some(mode) => {
                        self.set_scene_mode(mode);
                        self.console_feedback(format!("scene mode set to '{}'", mode.as_str()));
                    }
                    None => self.console_feedback("invalid scene mode (expected: 3d|2d)"),
                }
            }
            "tile.status" => {
                let t = self.tile_world_frame;
                self.console_feedback(format!(
                    "tile world | chunks={} dirty={} visible_chunks={} visible_tiles={} emitted_tiles={} culled_tiles={} world_rev={} mutations={}",
                    t.loaded_chunks,
                    t.dirty_chunks,
                    t.visible_chunks,
                    t.visible_tiles,
                    t.emitted_tiles,
                    t.culled_tiles,
                    t.world_revision,
                    t.mutation_count
                ));
            }
            "tile.set" => {
                let Some(raw_x) = parts.next() else {
                    self.console_feedback("usage: tile.set <x y id>");
                    return RuntimeCommand::Consumed;
                };
                let Some(raw_y) = parts.next() else {
                    self.console_feedback("usage: tile.set <x y id>");
                    return RuntimeCommand::Consumed;
                };
                let Some(raw_id) = parts.next() else {
                    self.console_feedback("usage: tile.set <x y id>");
                    return RuntimeCommand::Consumed;
                };
                let x = match parse_console_i32_in_range(raw_x, "tile.set.x", -262_144, 262_144) {
                    Ok(value) => value,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let y = match parse_console_i32_in_range(raw_y, "tile.set.y", -262_144, 262_144) {
                    Ok(value) => value,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let tile_id = match parse_console_u16_in_range(raw_id, "tile.set.id", 1, u16::MAX) {
                    Ok(value) => value,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let coord = TileCoord2d::new(x, y);
                let before = self.tile_world_2d.tile(coord);
                let changed = self.tile_world_2d.set_tile(coord, tile_id);
                self.tile_world_frame = self.tile_world_2d.telemetry_snapshot();
                self.console_feedback(format!(
                    "tile.set ({x},{y}) {} -> {}{}",
                    before,
                    tile_id,
                    if changed { "" } else { " (unchanged)" }
                ));
            }
            "tile.dig" => {
                let Some(raw_x) = parts.next() else {
                    self.console_feedback("usage: tile.dig <x y>");
                    return RuntimeCommand::Consumed;
                };
                let Some(raw_y) = parts.next() else {
                    self.console_feedback("usage: tile.dig <x y>");
                    return RuntimeCommand::Consumed;
                };
                let x = match parse_console_i32_in_range(raw_x, "tile.dig.x", -262_144, 262_144) {
                    Ok(value) => value,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let y = match parse_console_i32_in_range(raw_y, "tile.dig.y", -262_144, 262_144) {
                    Ok(value) => value,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let coord = TileCoord2d::new(x, y);
                let before = self.tile_world_2d.tile(coord);
                let changed = self
                    .tile_world_2d
                    .apply_mutation(TileMutation2d::dig(coord));
                self.tile_world_frame = self.tile_world_2d.telemetry_snapshot();
                self.console_feedback(format!(
                    "tile.dig ({x},{y}) {} -> 0{}",
                    before,
                    if changed { "" } else { " (unchanged)" }
                ));
            }
            "tile.fill" => {
                let Some(raw_x0) = parts.next() else {
                    self.console_feedback("usage: tile.fill <x0 y0 x1 y1 id>");
                    return RuntimeCommand::Consumed;
                };
                let Some(raw_y0) = parts.next() else {
                    self.console_feedback("usage: tile.fill <x0 y0 x1 y1 id>");
                    return RuntimeCommand::Consumed;
                };
                let Some(raw_x1) = parts.next() else {
                    self.console_feedback("usage: tile.fill <x0 y0 x1 y1 id>");
                    return RuntimeCommand::Consumed;
                };
                let Some(raw_y1) = parts.next() else {
                    self.console_feedback("usage: tile.fill <x0 y0 x1 y1 id>");
                    return RuntimeCommand::Consumed;
                };
                let Some(raw_id) = parts.next() else {
                    self.console_feedback("usage: tile.fill <x0 y0 x1 y1 id>");
                    return RuntimeCommand::Consumed;
                };
                let x0 = match parse_console_i32_in_range(raw_x0, "tile.fill.x0", -262_144, 262_144)
                {
                    Ok(value) => value,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let y0 = match parse_console_i32_in_range(raw_y0, "tile.fill.y0", -262_144, 262_144)
                {
                    Ok(value) => value,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let x1 = match parse_console_i32_in_range(raw_x1, "tile.fill.x1", -262_144, 262_144)
                {
                    Ok(value) => value,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let y1 = match parse_console_i32_in_range(raw_y1, "tile.fill.y1", -262_144, 262_144)
                {
                    Ok(value) => value,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let tile_id = match parse_console_u16_in_range(raw_id, "tile.fill.id", 0, u16::MAX)
                {
                    Ok(value) => value,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let changed = self.tile_world_2d.fill_rect(
                    TileCoord2d::new(x0, y0),
                    TileCoord2d::new(x1, y1),
                    tile_id,
                );
                self.tile_world_frame = self.tile_world_2d.telemetry_snapshot();
                self.console_feedback(format!(
                    "tile.fill [{x0},{y0}]..[{x1},{y1}] id={tile_id} changed={changed}"
                ));
            }
            "scene.reload" => {
                let mut notes = Vec::new();
                match self.reload_script_runtime_from_sources(true) {
                    Ok(note) => notes.push(note),
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                }
                if self.sprite_loader.is_some() {
                    match self.reload_sprite_from_watcher() {
                        Ok(note) => notes.push(note),
                        Err(err) => self.console_feedback(format!("sprite reload warning: {err}")),
                    }
                }
                self.reset_simulation_state();
                self.camera.reset_pose();
                self.console_feedback("scene state reset");
                for note in notes {
                    self.console_feedback(note);
                }
            }
            "script.reload" => match self.reload_script_runtime_from_sources(false) {
                Ok(note) => self.console_feedback(note),
                Err(err) => self.console_feedback(err),
            },
            "sprite.reload" => {
                if self.sprite_loader.is_some() {
                    match self.reload_sprite_from_watcher() {
                        Ok(note) => self.console_feedback(note),
                        Err(err) => self.console_feedback(err),
                    }
                } else if self.cli_options.project_path.is_some()
                    || self.cli_options.joint_path.is_some()
                {
                    match self.reload_script_runtime_from_sources(true) {
                        Ok(note) => self.console_feedback(format!(
                            "sprite refreshed via bundle reload: {note}"
                        )),
                        Err(err) => self.console_feedback(err),
                    }
                } else {
                    self.console_feedback(
                        "sprite.reload is unavailable (no sprite watcher / bundle context)",
                    );
                }
            }
            "perf.snapshot" => {
                let fps = self.fps_tracker.snapshot();
                let fsr = self.renderer.fsr_status();
                let rt = self.renderer.ray_tracing_status();
                let gravity = self.world.borrow().gravity();
                let perf_contract = self.performance_contract_evaluation();
                self.console_feedback(format!(
                    "perf snapshot | pipeline={} bridge_path={} q={} +{} -{} lag={} fallback={} totals[published={} drained={} popped={}] | fps inst={:.1} ema={:.1} avg={:.1} stddev={:.2}ms | frame_ema={:.2}ms jitter_ema={:.2}ms | fill={:.2}/{:.2} | balls={} target={} draw_limit={:?} | substeps={}/{} manual={:?} | grav=[{:.2},{:.2},{:.2}] | rt={:?}/{} dyn={} | fsr={:?}/{:?} scale={:.2} sharp={:.2} | {}",
                    self.pipeline_mode.as_str(),
                    self.runtime_bridge_telemetry.bridge_path.as_str(),
                    self.runtime_bridge_telemetry.queued_plan_depth,
                    self.runtime_bridge_telemetry.bridge_pump_published,
                    self.runtime_bridge_telemetry.bridge_pump_drained,
                    self.runtime_bridge_telemetry.physics_lag_frames,
                    self.runtime_bridge_telemetry.used_fallback_plan,
                    self.runtime_bridge_metrics.bridge_tick_published_frames,
                    self.runtime_bridge_metrics.bridge_tick_drained_plans,
                    self.runtime_bridge_metrics.frame_plans_popped,
                    fps.instant_fps,
                    fps.ema_fps,
                    fps.avg_fps,
                    fps.frame_time_stddev_ms,
                    self.frame_time_ema_ms,
                    self.frame_time_jitter_ema_ms,
                    self.last_framebuffer_fill_ratio,
                    self.framebuffer_fill_ema,
                    self.scene.live_ball_count(),
                    self.scene.config().target_ball_count,
                    self.adaptive_ball_render_limit,
                    self.last_substeps,
                    self.max_substeps,
                    self.manual_max_substeps,
                    gravity.x,
                    gravity.y,
                    gravity.z,
                    self.rt_mode,
                    if rt.active { "on" } else { "off" },
                    rt.rt_dynamic_count,
                    fsr.requested_mode,
                    if fsr.active { "on" } else { "off" },
                    fsr.render_scale,
                    fsr.sharpness,
                    perf_contract.compact_summary()
                ));
            }
            "perf.contract" => {
                let requested_scenario = match parts.next() {
                    Some(raw) => match PerformanceContractScenario::parse(raw) {
                        Some(scenario) => Some(scenario),
                        None => {
                            self.console_feedback("usage: perf.contract [8k|30k|60k]");
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => None,
                };
                for line in self
                    .performance_contract_evaluation_for(requested_scenario)
                    .detail_lines()
                {
                    self.console_feedback(line);
                }
            }
            "perf.report" => {
                let requested_scenario = match parts.next() {
                    Some(raw) => match PerformanceContractScenario::parse(raw) {
                        Some(scenario) => Some(scenario),
                        None => {
                            self.console_feedback("usage: perf.report [8k|30k|60k]");
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => None,
                };
                let evaluation = self.performance_contract_evaluation_for(requested_scenario);
                let fps = self.fps_tracker.snapshot();
                let config = self.scene.config();
                let fps_cap = self
                    .frame_cap_interval
                    .map(|interval| format!("{:.0}", 1.0 / interval.as_secs_f32().max(1e-6)))
                    .unwrap_or_else(|| "off".to_string());
                let substeps = self
                    .manual_max_substeps
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| format!("auto:{}", self.max_substeps));
                let render_distance = self
                    .render_distance
                    .map(|value| format!("{value:.1}"))
                    .unwrap_or_else(|| "off".to_string());
                let rt_status = self.renderer.ray_tracing_status();
                self.console_feedback(format!(
                    "perf report | scenario={} tier={} stable={} live={} target={} spawn_per_tick={} tick_profile={:?} substeps={} fps_cap={} vsync={:?}",
                    evaluation.scenario.label(),
                    evaluation.tier.label(),
                    evaluation.stable,
                    evaluation.live_balls,
                    config.target_ball_count,
                    config.spawn_per_tick,
                    self.tick_profile,
                    substeps,
                    fps_cap,
                    self.present_mode,
                ));
                self.console_feedback(format!(
                    "  timing | fps inst={:.1} ema={:.1} avg={:.1} tick={:.0}Hz stddev={:.2}ms jitter={:.2}ms lag={} phys_us={}",
                    fps.instant_fps,
                    fps.ema_fps,
                    fps.avg_fps,
                    self.tick_hz,
                    fps.frame_time_stddev_ms,
                    self.frame_time_jitter_ema_ms,
                    self.runtime_bridge_telemetry.physics_lag_frames,
                    self.world.borrow().last_step_timings.total_us(),
                ));
                self.console_feedback(format!(
                    "  render | backend={} scheduler={} pipeline={} render_distance={} adaptive_distance={} distance_blur={:?} fsr={:?}/{:?} scale={:.2} sharpness={:.2} rt={:?}/{} dyn={} adapter={}",
                    self.renderer.backend_label(),
                    scheduler_path_label(self.scheduler_path),
                    self.pipeline_mode.as_str(),
                    render_distance,
                    self.adaptive_distance_enabled,
                    self.distance_blur_mode,
                    self.fsr_config.mode,
                    self.fsr_config.quality,
                    self.fsr_dynamo_scale,
                    self.fsr_config.sharpness,
                    self.rt_mode,
                    rt_status.active,
                    rt_status.rt_dynamic_count,
                    self.adapter_name,
                ));
                for line in evaluation.detail_lines() {
                    self.console_feedback(line);
                }
            }
            "perf.preset" => {
                let Some(raw) = parts.next() else {
                    self.console_feedback("usage: perf.preset <8k|30k|60k>");
                    return RuntimeCommand::Consumed;
                };
                let Some(scenario) = PerformanceContractScenario::parse(raw) else {
                    self.console_feedback("usage: perf.preset <8k|30k|60k>");
                    return RuntimeCommand::Consumed;
                };
                match self.apply_performance_contract_preset(scenario) {
                    Ok(note) => {
                        self.console_feedback(note);
                        let target = performance_contract_target(scenario);
                        self.console_feedback(format!(
                            "  preset target | ship fps>={:.1} tick>={:.0}Hz stddev<={:.1}ms jitter<={:.1}ms | stretch fps>={:.1} tick>={:.0}Hz stddev<={:.1}ms jitter<={:.1}ms",
                            target.ship_fps_min,
                            target.ship_tick_min,
                            target.ship_stddev_ms_max,
                            target.ship_jitter_ms_max,
                            target.stretch_fps_min,
                            target.stretch_tick_min,
                            target.stretch_stddev_ms_max,
                            target.stretch_jitter_ms_max,
                        ));
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "phys.gravity" => {
                let Some(x_raw) = parts.next() else {
                    self.console_feedback("usage: phys.gravity <x> <y> <z>");
                    return RuntimeCommand::Consumed;
                };
                let Some(y_raw) = parts.next() else {
                    self.console_feedback("usage: phys.gravity <x> <y> <z>");
                    return RuntimeCommand::Consumed;
                };
                let Some(z_raw) = parts.next() else {
                    self.console_feedback("usage: phys.gravity <x> <y> <z>");
                    return RuntimeCommand::Consumed;
                };
                let x = match parse_console_f32_in_range(x_raw, "phys.gravity.x", -120.0, 120.0) {
                    Ok(v) => v,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let y = match parse_console_f32_in_range(y_raw, "phys.gravity.y", -120.0, 120.0) {
                    Ok(v) => v,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let z = match parse_console_f32_in_range(z_raw, "phys.gravity.z", -120.0, 120.0) {
                    Ok(v) => v,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                self.world.borrow_mut().set_gravity(Vector3::new(x, y, z));
                self.console_feedback(format!("gravity set to [{x:.3}, {y:.3}, {z:.3}]"));
            }
            "phys.substeps" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: phys.substeps <auto|n>");
                    return RuntimeCommand::Consumed;
                };
                if value.eq_ignore_ascii_case("auto") {
                    self.manual_max_substeps = None;
                    self.tick_retune_timer = 0.0;
                    self.console_feedback("substeps override cleared (auto)");
                } else {
                    let substeps = match parse_console_u32_in_range(value, "phys.substeps", 1, 64) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    };
                    self.manual_max_substeps = Some(substeps);
                    self.max_substeps = substeps;
                    self.world
                        .borrow_mut()
                        .set_timestep(1.0 / self.tick_hz.max(1.0), self.max_substeps);
                    self.console_feedback(format!("manual max_substeps={substeps}"));
                }
            }
            "cam.speed" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: cam.speed <1..200>");
                    return RuntimeCommand::Consumed;
                };
                match parse_console_f32_in_range(value, "cam.speed", 1.0, 200.0) {
                    Ok(speed) => {
                        self.camera.set_move_speed(speed);
                        self.console_feedback(format!("camera speed set to {speed:.2}"));
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "cam.sens" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: cam.sens <0.0001..0.02>");
                    return RuntimeCommand::Consumed;
                };
                match parse_console_f32_in_range(value, "cam.sens", 0.0001, 0.02) {
                    Ok(sens) => {
                        self.camera.set_mouse_sensitivity(sens);
                        self.console_feedback(format!("camera sensitivity set to {sens:.5}"));
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "cam.reset" => {
                self.camera.reset_pose();
                self.console_feedback("camera pose reset");
            }
            "log.clear" => {
                self.console.log_lines.clear();
                self.console.log_scroll = 0;
                self.console_feedback("log buffer cleared");
            }
            "log.tail" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: log.tail <off|n>");
                    return RuntimeCommand::Consumed;
                };
                if value.eq_ignore_ascii_case("off") {
                    self.console.log_tail_limit = None;
                    self.console.log_scroll = 0;
                    self.console_feedback("log tail disabled");
                } else {
                    match parse_console_u32_in_range(value, "log.tail", 1, 320) {
                        Ok(limit) => {
                            self.console.log_tail_limit = Some(limit as usize);
                            self.console.log_scroll =
                                self.console.log_scroll.min(self.max_console_log_scroll());
                            self.console_feedback(format!("log tail set to last {limit} lines"));
                        }
                        Err(err) => self.console_feedback(err),
                    }
                }
            }
            "log.level" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: log.level <all|info|error>");
                    return RuntimeCommand::Consumed;
                };
                let next = if value.eq_ignore_ascii_case("all") {
                    Some(RuntimeConsoleLogFilter::All)
                } else if value.eq_ignore_ascii_case("info") {
                    Some(RuntimeConsoleLogFilter::Info)
                } else if value.eq_ignore_ascii_case("error") {
                    Some(RuntimeConsoleLogFilter::Error)
                } else {
                    None
                };
                match next {
                    Some(filter) => {
                        self.console.log_filter = filter;
                        self.console.log_scroll =
                            self.console.log_scroll.min(self.max_console_log_scroll());
                        self.console_feedback(format!(
                            "log filter set to {}",
                            value.to_ascii_lowercase()
                        ));
                    }
                    None => self.console_feedback("usage: log.level <all|info|error>"),
                }
            }
            "file.exists" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback("usage: file.exists <path>");
                    return RuntimeCommand::Consumed;
                };
                let candidate = match self.resolve_console_candidate_path(path_raw) {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !candidate.exists() {
                    self.console_feedback(format!("file.exists '{path_raw}' => false"));
                    return RuntimeCommand::Consumed;
                }
                match candidate.canonicalize() {
                    Ok(canonical) => {
                        if !canonical.starts_with(&self.file_io_root) {
                            self.console_feedback(format!(
                                "file.exists denied: '{}' is outside workspace root '{}'",
                                canonical.display(),
                                self.file_io_root.display()
                            ));
                        } else {
                            self.console_feedback(format!(
                                "file.exists '{path_raw}' => true ({})",
                                canonical.display()
                            ));
                        }
                    }
                    Err(err) => {
                        self.console_feedback(format!("file.exists failed for '{path_raw}': {err}"))
                    }
                }
            }
            "file.head" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback("usage: file.head <path> [lines]");
                    return RuntimeCommand::Consumed;
                };
                let line_limit = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.head.lines",
                        1,
                        CONSOLE_FILE_HEAD_MAX_LINES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_HEAD_DEFAULT_LINES,
                };
                let path = match self.resolve_console_existing_path(path_raw, "file.head") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!("file.head: '{}' is not a file", path.display()));
                    return RuntimeCommand::Consumed;
                }
                match self.read_file_prefix_bytes(&path, CONSOLE_FILE_HEAD_MAX_WINDOW_BYTES) {
                    Ok((bytes, truncated)) => {
                        self.console_feedback(format!(
                            "file.head '{}' ({} byte(s) window, lines={line_limit}{})",
                            path.display(),
                            bytes.len(),
                            if truncated { ", truncated window" } else { "" }
                        ));
                        let text = String::from_utf8_lossy(&bytes);
                        self.emit_console_text_lines("  ", &text, line_limit);
                    }
                    Err(err) => self.console_feedback(format!("file.head: {err}")),
                }
            }
            "file.tail" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback("usage: file.tail <path> [lines] [max_bytes]");
                    return RuntimeCommand::Consumed;
                };
                let line_limit = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.tail.lines",
                        1,
                        CONSOLE_FILE_TAIL_MAX_LINES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_TAIL_DEFAULT_LINES,
                };
                let max_window_bytes = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.tail.max_bytes",
                        64,
                        CONSOLE_FILE_TAIL_MAX_WINDOW_BYTES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_TAIL_DEFAULT_WINDOW_BYTES,
                };
                let path = match self.resolve_console_existing_path(path_raw, "file.tail") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!("file.tail: '{}' is not a file", path.display()));
                    return RuntimeCommand::Consumed;
                }
                match self.read_file_tail_window(&path, max_window_bytes) {
                    Ok((bytes, truncated_window)) => {
                        let text = String::from_utf8_lossy(&bytes);
                        let lines = text.lines().collect::<Vec<_>>();
                        self.console_feedback(format!(
                            "file.tail '{}' ({} byte(s) window, lines={line_limit}{})",
                            path.display(),
                            bytes.len(),
                            if truncated_window {
                                ", truncated window"
                            } else {
                                ""
                            }
                        ));
                        if lines.is_empty() {
                            self.console_feedback("  <empty>");
                            return RuntimeCommand::Consumed;
                        }
                        let start = lines.len().saturating_sub(line_limit);
                        for line in lines.iter().skip(start) {
                            self.console_feedback(format!("  {line}"));
                        }
                        if lines.len() > line_limit {
                            self.console_feedback(format!(
                                "  ... showing last {line_limit} line(s) from window"
                            ));
                        }
                    }
                    Err(err) => self.console_feedback(format!("file.tail: {err}")),
                }
            }
            "file.find" | "file.findi" => {
                let case_insensitive = matches!(head_lc.as_str(), "file.findi");
                let Some(path_raw) = parts.next() else {
                    self.console_feedback(
                        "usage: file.find <path> <pattern> [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let Some(pattern) = parts.next() else {
                    self.console_feedback(
                        "usage: file.find <path> <pattern> [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let max_matches = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        if case_insensitive {
                            "file.findi.max_matches"
                        } else {
                            "file.find.max_matches"
                        },
                        1,
                        CONSOLE_FILE_FIND_MAX_MATCHES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_MATCHES,
                };
                let max_bytes = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        if case_insensitive {
                            "file.findi.max_bytes"
                        } else {
                            "file.find.max_bytes"
                        },
                        64,
                        CONSOLE_FILE_FIND_MAX_BYTES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_BYTES,
                };
                let path = match self.resolve_console_existing_path(
                    path_raw,
                    if case_insensitive {
                        "file.findi"
                    } else {
                        "file.find"
                    },
                ) {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!(
                        "{}: '{}' is not a file",
                        if case_insensitive {
                            "file.findi"
                        } else {
                            "file.find"
                        },
                        path.display()
                    ));
                    return RuntimeCommand::Consumed;
                }
                self.run_file_find(&path, pattern, max_matches, max_bytes, case_insensitive);
            }
            "file.findr" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback(
                        "usage: file.findr <path> <regex> [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let Some(pattern) = parts.next() else {
                    self.console_feedback(
                        "usage: file.findr <path> <regex> [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let max_matches = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.findr.max_matches",
                        1,
                        CONSOLE_FILE_FIND_MAX_MATCHES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_MATCHES,
                };
                let max_bytes = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.findr.max_bytes",
                        64,
                        CONSOLE_FILE_FIND_MAX_BYTES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_BYTES,
                };
                let path = match self.resolve_console_existing_path(path_raw, "file.findr") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!(
                        "file.findr: '{}' is not a file",
                        path.display()
                    ));
                    return RuntimeCommand::Consumed;
                }
                self.run_file_find_regex(&path, pattern, max_matches, max_bytes);
            }
            "file.grep" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback(
                        "usage: file.grep <path> <pattern> [context] [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let Some(pattern) = parts.next() else {
                    self.console_feedback(
                        "usage: file.grep <path> <pattern> [context] [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let context = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.grep.context",
                        0,
                        CONSOLE_FILE_GREP_MAX_CONTEXT,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_GREP_DEFAULT_CONTEXT,
                };
                let max_matches = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.grep.max_matches",
                        1,
                        CONSOLE_FILE_FIND_MAX_MATCHES,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_MATCHES,
                };
                let max_bytes = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.grep.max_bytes",
                        64,
                        CONSOLE_FILE_FIND_MAX_BYTES,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_BYTES,
                };
                let path = match self.resolve_console_existing_path(path_raw, "file.grep") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!("file.grep: '{}' is not a file", path.display()));
                    return RuntimeCommand::Consumed;
                }
                self.run_file_grep(&path, pattern, context, max_matches, max_bytes);
            }
            "file.tailf" => {
                let Some(arg) = parts.next() else {
                    if let Some(follow) = self.console.tail_follow.as_ref() {
                        self.console_feedback(format!(
                            "tailf active path='{}' poll={}ms lines/poll={}",
                            follow.path.display(),
                            follow.poll_interval.as_millis(),
                            follow.max_lines_per_poll
                        ));
                    } else {
                        self.console_feedback(
                            "usage: file.tailf <path>|stop [poll_ms] [max_lines]",
                        );
                    }
                    return RuntimeCommand::Consumed;
                };
                if arg.eq_ignore_ascii_case("stop") || arg.eq_ignore_ascii_case("off") {
                    if let Some(prev) = self.console.tail_follow.take() {
                        self.console_feedback(format!(
                            "tailf stopped for '{}'",
                            prev.path.display()
                        ));
                    } else {
                        self.console_feedback("tailf already inactive");
                    }
                    return RuntimeCommand::Consumed;
                }

                let poll_ms = match parts.next() {
                    Some(raw) => {
                        match parse_console_usize_in_range(raw, "file.tailf.poll_ms", 50, 10_000) {
                            Ok(v) => v as u64,
                            Err(err) => {
                                self.console_feedback(err);
                                return RuntimeCommand::Consumed;
                            }
                        }
                    }
                    None => CONSOLE_FILE_TAILF_DEFAULT_POLL_MS,
                };
                let max_lines_per_poll = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.tailf.max_lines",
                        1,
                        CONSOLE_FILE_TAILF_MAX_LINES_PER_POLL,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_TAILF_MAX_LINES_PER_POLL / 2,
                };
                let path = match self.resolve_console_existing_path(arg, "file.tailf") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!(
                        "file.tailf: '{}' is not a file",
                        path.display()
                    ));
                    return RuntimeCommand::Consumed;
                }
                let start_cursor = match fs::metadata(&path) {
                    Ok(meta) => meta.len(),
                    Err(err) => {
                        self.console_feedback(format!(
                            "file.tailf: failed to read metadata '{}': {err}",
                            path.display()
                        ));
                        return RuntimeCommand::Consumed;
                    }
                };
                self.console.tail_follow = Some(RuntimeConsoleTailFollow {
                    path: path.clone(),
                    poll_interval: Duration::from_millis(poll_ms),
                    max_lines_per_poll,
                    cursor: start_cursor,
                    partial_line: String::new(),
                    last_poll: Instant::now(),
                    last_error: None,
                });
                self.console_feedback(format!(
                    "tailf started for '{}' poll={}ms max_lines={}",
                    path.display(),
                    poll_ms,
                    max_lines_per_poll
                ));
            }
            "file.watch" => {
                let Some(arg) = parts.next() else {
                    if let Some(watch) = self.console.file_watch.as_ref() {
                        self.console_feedback(format!(
                            "watch active path='{}' poll={}ms size={}",
                            watch.path.display(),
                            watch.poll_interval.as_millis(),
                            watch.last_len
                        ));
                    } else {
                        self.console_feedback("usage: file.watch <path>|stop [poll_ms]");
                    }
                    return RuntimeCommand::Consumed;
                };
                if arg.eq_ignore_ascii_case("stop") || arg.eq_ignore_ascii_case("off") {
                    if let Some(prev) = self.console.file_watch.take() {
                        self.console_feedback(format!(
                            "watch stopped for '{}'",
                            prev.path.display()
                        ));
                    } else {
                        self.console_feedback("watch already inactive");
                    }
                    return RuntimeCommand::Consumed;
                }

                let poll_ms = match parts.next() {
                    Some(raw) => {
                        match parse_console_usize_in_range(raw, "file.watch.poll_ms", 50, 10_000) {
                            Ok(v) => v as u64,
                            Err(err) => {
                                self.console_feedback(err);
                                return RuntimeCommand::Consumed;
                            }
                        }
                    }
                    None => CONSOLE_FILE_WATCH_DEFAULT_POLL_MS,
                };
                let path = match self.resolve_console_existing_path(arg, "file.watch") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!(
                        "file.watch: '{}' is not a file",
                        path.display()
                    ));
                    return RuntimeCommand::Consumed;
                }
                let (modified, len) = match fs::metadata(&path) {
                    Ok(meta) => (meta.modified().ok(), meta.len()),
                    Err(err) => {
                        self.console_feedback(format!(
                            "file.watch: failed to read metadata '{}': {err}",
                            path.display()
                        ));
                        return RuntimeCommand::Consumed;
                    }
                };
                self.console.file_watch = Some(RuntimeConsoleFileWatch {
                    path: path.clone(),
                    poll_interval: Duration::from_millis(poll_ms),
                    last_modified: modified,
                    last_len: len,
                    last_poll: Instant::now(),
                    last_error: None,
                });
                self.console_feedback(format!(
                    "watch started for '{}' poll={}ms baseline_size={}",
                    path.display(),
                    poll_ms,
                    len
                ));
            }
            "file.read" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback("usage: file.read <path> [max_bytes]");
                    return RuntimeCommand::Consumed;
                };
                let max_bytes = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.read.max_bytes",
                        64,
                        CONSOLE_FILE_READ_MAX_BYTES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_READ_DEFAULT_BYTES,
                };
                let path = match self.resolve_console_existing_path(path_raw, "file.read") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!("file.read: '{}' is not a file", path.display()));
                    return RuntimeCommand::Consumed;
                }
                match self.read_file_prefix_bytes(&path, max_bytes) {
                    Ok((buf, truncated)) => {
                        let preview = String::from_utf8_lossy(&buf);
                        self.console_feedback(format!(
                            "file.read '{}' (preview {} byte(s), limit {max_bytes}{})",
                            path.display(),
                            buf.len(),
                            if truncated { ", truncated" } else { "" }
                        ));
                        self.emit_console_text_lines(
                            "  ",
                            &preview,
                            CONSOLE_FILE_HEAD_DEFAULT_LINES,
                        );
                    }
                    Err(err) => self.console_feedback(format!("file.read: {err}")),
                }
            }
            "file.list" => {
                let dir_raw = parts.next().unwrap_or(".");
                let limit = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.list.limit",
                        1,
                        CONSOLE_FILE_LIST_MAX_LIMIT,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_LIST_DEFAULT_LIMIT,
                };
                let dir = match self.resolve_console_existing_path(dir_raw, "file.list") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !dir.is_dir() {
                    self.console_feedback(format!(
                        "file.list: '{}' is not a directory",
                        dir.display()
                    ));
                    return RuntimeCommand::Consumed;
                }
                let mut entries = match fs::read_dir(&dir) {
                    Ok(read_dir) => read_dir
                        .filter_map(|entry| entry.ok())
                        .map(|entry| {
                            let path = entry.path();
                            let name = entry.file_name().to_string_lossy().to_string();
                            let is_dir = path.is_dir();
                            (name, is_dir)
                        })
                        .collect::<Vec<_>>(),
                    Err(err) => {
                        self.console_feedback(format!(
                            "file.list failed for '{}': {err}",
                            dir.display()
                        ));
                        return RuntimeCommand::Consumed;
                    }
                };
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                self.console_feedback(format!(
                    "file.list '{}' total={} showing={}",
                    dir.display(),
                    entries.len(),
                    entries.len().min(limit)
                ));
                for (name, is_dir) in entries.iter().take(limit) {
                    self.console_feedback(format!(
                        "  [{}] {}",
                        if *is_dir { "d" } else { "f" },
                        name
                    ));
                }
                if entries.len() > limit {
                    self.console_feedback(format!(
                        "  ... {} more entrie(s) hidden",
                        entries.len() - limit
                    ));
                }
            }
            "clear" => {
                self.console.last_feedback.clear();
                self.console_feedback("console status cleared");
            }
            "exit" | "quit" => {
                self.console_feedback("exit requested from console");
                return RuntimeCommand::Exit;
            }
            "gfx.vsync" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.vsync <auto|on|off>");
                    return RuntimeCommand::Consumed;
                };
                match VsyncMode::parse(value) {
                    Ok(mode) => {
                        let present_mode = match mode {
                            VsyncMode::Auto => wgpu::PresentMode::AutoVsync,
                            VsyncMode::On => wgpu::PresentMode::Fifo,
                            VsyncMode::Off => wgpu::PresentMode::AutoNoVsync,
                        };
                        self.apply_present_mode(present_mode);
                        self.console_feedback(format!("vsync set to {mode:?} ({present_mode:?})"));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.fps_cap" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.fps_cap <off|N>");
                    return RuntimeCommand::Consumed;
                };
                match parse_fps_cap(value) {
                    Ok(cap) => {
                        self.apply_fps_cap_runtime(cap);
                        if let Some(fps) = cap {
                            self.console_feedback(format!("fps cap set to {fps:.1}"));
                        } else {
                            self.console_feedback("fps cap disabled".to_string());
                        }
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.rt" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.rt <off|auto|on>");
                    return RuntimeCommand::Consumed;
                };
                let Some(mode) = RayTracingMode::from_str(value) else {
                    self.console_feedback("invalid gfx.rt value (expected off|auto|on)");
                    return RuntimeCommand::Consumed;
                };
                self.rt_mode = mode;
                self.renderer.set_ray_tracing_mode(&self.queue, mode);
                let status = self.renderer.ray_tracing_status();
                self.console_feedback(format!(
                    "rt set to {:?} (active={}, reason={})",
                    mode,
                    status.active,
                    if status.fallback_reason.is_empty() {
                        "none"
                    } else {
                        status.fallback_reason.as_str()
                    }
                ));
            }
            "gfx.fsr" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.fsr <off|auto|on>");
                    return RuntimeCommand::Consumed;
                };
                let Some(mode) = FsrMode::parse(value) else {
                    self.console_feedback("invalid gfx.fsr value (expected off|auto|on)");
                    return RuntimeCommand::Consumed;
                };
                self.fsr_config.mode = mode;
                self.renderer.set_fsr_config(&self.queue, self.fsr_config);
                self.console_feedback(format!("fsr mode set to {:?}", mode));
            }
            "gfx.fsr_quality" => {
                let Some(value) = parts.next() else {
                    self.console_feedback(
                        "usage: gfx.fsr_quality <native|ultra|quality|balanced|performance>",
                    );
                    return RuntimeCommand::Consumed;
                };
                let Some(quality) = FsrQualityPreset::parse(value) else {
                    self.console_feedback("invalid gfx.fsr_quality preset");
                    return RuntimeCommand::Consumed;
                };
                self.fsr_config.quality = quality;
                self.renderer.set_fsr_config(&self.queue, self.fsr_config);
                self.console_feedback(format!("fsr quality set to {:?}", quality));
            }
            "gfx.fsr_sharpness" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.fsr_sharpness <0..1>");
                    return RuntimeCommand::Consumed;
                };
                match parse_fsr_sharpness(value) {
                    Ok(sharpness) => {
                        self.fsr_config.sharpness = sharpness;
                        self.renderer.set_fsr_config(&self.queue, self.fsr_config);
                        self.console_feedback(format!("fsr sharpness set to {sharpness:.2}"));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.msaa" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.msaa <off|4>");
                    return RuntimeCommand::Consumed;
                };
                match parse_msaa(value) {
                    Ok(count) => {
                        self.renderer.set_msaa_sample_count(&self.device, count);
                        let label = if count > 1 {
                            format!("msaa {count}x")
                        } else {
                            "msaa off".to_string()
                        };
                        self.console_feedback(format!("{label} applied"));
                        self.console.quick_msaa = if count > 1 {
                            format!("{count}")
                        } else {
                            "off".to_string()
                        };
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.fsr_scale" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.fsr_scale <auto|0.5..1>");
                    return RuntimeCommand::Consumed;
                };
                match parse_fsr_scale(value) {
                    Ok(scale) => {
                        self.fsr_config.render_scale_override = scale;
                        self.renderer.set_fsr_config(&self.queue, self.fsr_config);
                        match scale {
                            Some(v) => self.console_feedback(format!("fsr scale override={v:.2}")),
                            None => self.console_feedback("fsr scale override=auto"),
                        }
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.profile" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.profile <low|med|high|ultra>");
                    return RuntimeCommand::Consumed;
                };
                let profile = value.to_ascii_lowercase();
                match self.apply_gfx_profile(profile.as_str()) {
                    Ok(note) => self.console_feedback(note),
                    Err(err) => self.console_feedback(err),
                }
            }
            "gfx.render_distance" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.render_distance <off|N>");
                    return RuntimeCommand::Consumed;
                };
                match parse_render_distance(value) {
                    Ok(distance) => {
                        self.render_distance = distance;
                        if let Some(value) = distance {
                            self.render_distance_min = (value * 0.72).clamp(28.0, value);
                            self.render_distance_max =
                                (value * 1.55).clamp(value, value.max(220.0));
                            self.console_feedback(format!("render distance set to {value:.1}"));
                        } else {
                            self.console_feedback("render distance disabled");
                        }
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.adaptive_distance" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.adaptive_distance <auto|on|off>");
                    return RuntimeCommand::Consumed;
                };
                match ToggleAuto::parse(value, "gfx.adaptive_distance") {
                    Ok(mode) => {
                        self.adaptive_distance_enabled = mode.resolve(true);
                        self.console_feedback(format!(
                            "adaptive distance mode={mode:?} enabled={}",
                            self.adaptive_distance_enabled
                        ));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.distance_blur" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.distance_blur <auto|on|off>");
                    return RuntimeCommand::Consumed;
                };
                match ToggleAuto::parse(value, "gfx.distance_blur") {
                    Ok(mode) => {
                        self.distance_blur_mode = mode;
                        self.distance_blur_enabled = mode.resolve(false);
                        self.console_feedback(format!(
                            "distance blur mode={mode:?} enabled={}",
                            self.distance_blur_enabled
                        ));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "script.var" => {
                let mut tokens = trimmed.splitn(3, char::is_whitespace);
                let _ = tokens.next();
                let Some(name) = tokens.next() else {
                    self.console_feedback("usage: script.var <name> <expr>");
                    return RuntimeCommand::Consumed;
                };
                if !is_valid_tlscript_ident(name) {
                    self.console_feedback(format!("invalid variable name '{name}'"));
                    return RuntimeCommand::Consumed;
                }
                let value = tokens.next().unwrap_or("").trim();
                if value.is_empty() {
                    self.console_feedback("usage: script.var <name> <expr>");
                    return RuntimeCommand::Consumed;
                }
                self.console
                    .script_vars
                    .insert(name.to_string(), value.to_string());
                match self.rebuild_console_script_overlay() {
                    Ok(notes) => {
                        self.console_feedback(format!("script var '{name}' set to '{value}'"));
                        for note in notes {
                            self.console_feedback(note);
                        }
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "script.unset" => {
                let Some(name) = parts.next() else {
                    self.console_feedback("usage: script.unset <name>");
                    return RuntimeCommand::Consumed;
                };
                self.console.script_vars.remove(name);
                match self.rebuild_console_script_overlay() {
                    Ok(notes) => {
                        self.console_feedback(format!("script var '{name}' removed"));
                        for note in notes {
                            self.console_feedback(note);
                        }
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "script.vars" => {
                if self.console.script_vars.is_empty() {
                    self.console_feedback("script vars: <empty>");
                } else {
                    let vars = self
                        .console
                        .script_vars
                        .iter()
                        .map(|(k, v)| format!("{k}={v}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.console_feedback(format!("script vars: {vars}"));
                }
            }
            "script.call" | "script.exec" => {
                let statement_raw = trimmed
                    .split_once(char::is_whitespace)
                    .map(|(_, tail)| tail.trim())
                    .unwrap_or("");
                if statement_raw.is_empty() {
                    self.console_feedback(format!("usage: {head_lc} <statement>"));
                    return RuntimeCommand::Consumed;
                }
                let statement_template = if head_lc == "script.call" {
                    match Self::normalize_script_call_statement(statement_raw) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    }
                } else {
                    statement_raw.trim_end_matches(';').to_string()
                };
                self.console
                    .script_statements
                    .push(statement_template.clone());
                match self.rebuild_console_script_overlay() {
                    Ok(notes) => {
                        self.console_feedback(format!(
                            "script statement added [{}]: {}",
                            self.console.script_statements.len() - 1,
                            statement_template
                        ));
                        for note in notes {
                            self.console_feedback(note);
                        }
                    }
                    Err(err) => {
                        let _ = self.console.script_statements.pop();
                        self.console_feedback(err);
                    }
                }
            }
            "script.uncall" => {
                let Some(target) = parts.next() else {
                    self.console_feedback("usage: script.uncall <index|all>");
                    return RuntimeCommand::Consumed;
                };
                if target.eq_ignore_ascii_case("all") {
                    self.console.script_statements.clear();
                    self.console.script_overlay = empty_showcase_output();
                    self.console_feedback("all script statements removed");
                    return RuntimeCommand::Consumed;
                }
                let Ok(index) = target.parse::<usize>() else {
                    self.console_feedback("script.uncall expects numeric index or 'all'");
                    return RuntimeCommand::Consumed;
                };
                if index >= self.console.script_statements.len() {
                    self.console_feedback(format!(
                        "script.uncall index out of range (0..{})",
                        self.console.script_statements.len().saturating_sub(1)
                    ));
                    return RuntimeCommand::Consumed;
                }
                let removed = self.console.script_statements.remove(index);
                match self.rebuild_console_script_overlay() {
                    Ok(notes) => {
                        self.console_feedback(format!(
                            "removed script statement[{index}]: {removed}"
                        ));
                        for note in notes {
                            self.console_feedback(note);
                        }
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "script.list" => {
                if self.console.script_statements.is_empty() {
                    self.console_feedback("script statements: <empty>");
                } else {
                    let statements = self.console.script_statements.clone();
                    for (index, statement) in statements.iter().enumerate() {
                        self.console_feedback(format!("[{index}] {statement}"));
                    }
                }
            }
            "script.clear" => {
                self.console.script_vars.clear();
                self.console.script_statements.clear();
                self.console.script_overlay = empty_showcase_output();
                self.console_feedback("script vars and statements cleared");
            }
            _ => {
                self.console_feedback(format!("unknown command '{head}' (run 'help')"));
            }
        }
        RuntimeCommand::Consumed
    }

    pub(super) fn on_console_keyboard_input(&mut self, event: &KeyEvent) -> RuntimeCommand {
        if event.state != ElementState::Pressed {
            return RuntimeCommand::Consumed;
        }

        match event.physical_key {
            PhysicalKey::Code(KeyCode::Escape) => {
                self.toggle_console();
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::Tab) => {
                self.console.edit_target = self.console.edit_target.next();
                self.console_feedback(format!(
                    "active edit box: {}",
                    self.console.edit_target.label()
                ));
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::Enter) | PhysicalKey::Code(KeyCode::NumpadEnter) => {
                return self.apply_active_console_edit_target();
            }
            PhysicalKey::Code(KeyCode::Backspace) => {
                self.selected_console_edit_buffer_mut().pop();
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::ArrowUp) => {
                if self.console.edit_target != RuntimeConsoleEditTarget::Command {
                    return RuntimeCommand::Consumed;
                }
                if self.console.history.is_empty() {
                    return RuntimeCommand::Consumed;
                }
                let next = match self.console.history_cursor {
                    None => self.console.history.len().saturating_sub(1),
                    Some(cur) => cur.saturating_sub(1),
                };
                self.console.history_cursor = Some(next);
                self.console.input_line = self.console.history[next].clone();
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::ArrowDown) => {
                if self.console.edit_target != RuntimeConsoleEditTarget::Command {
                    return RuntimeCommand::Consumed;
                }
                if self.console.history.is_empty() {
                    return RuntimeCommand::Consumed;
                }
                let Some(cur) = self.console.history_cursor else {
                    return RuntimeCommand::Consumed;
                };
                if cur + 1 >= self.console.history.len() {
                    self.console.history_cursor = None;
                    self.console.input_line.clear();
                } else {
                    let next = cur + 1;
                    self.console.history_cursor = Some(next);
                    self.console.input_line = self.console.history[next].clone();
                }
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::PageUp) => {
                self.scroll_console_logs(4);
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::PageDown) => {
                self.scroll_console_logs(-4);
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::Home) => {
                self.console.log_scroll = self.max_console_log_scroll();
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::End) => {
                self.console.log_scroll = 0;
                return RuntimeCommand::Consumed;
            }
            _ => {}
        }

        if self.keyboard_modifiers.control_key()
            || self.keyboard_modifiers.alt_key()
            || self.keyboard_modifiers.super_key()
        {
            return RuntimeCommand::Consumed;
        }

        if let Some(text) = &event.text {
            for ch in text.chars() {
                if !ch.is_control() {
                    self.selected_console_edit_buffer_mut().push(ch);
                }
            }
        }
        RuntimeCommand::Consumed
    }

    pub(super) fn console_glyph_slot(ch: char) -> u16 {
        let code = ch as u32;
        if (32..=126).contains(&code) {
            CONSOLE_TEXT_SLOT_BASE + (code as u16 - 32)
        } else {
            CONSOLE_TEXT_SLOT_BASE + (u16::from(b'?') - 32)
        }
    }

    pub(super) fn push_console_text_line(
        sprites: &mut Vec<SpriteInstance>,
        sprite_id_seed: &mut u64,
        text: &str,
        x: f32,
        y: f32,
        z: f32,
        glyph_size: [f32; 2],
        color: [f32; 4],
        layer: i16,
    ) {
        let mut cursor_x = x;
        for ch in text.chars() {
            if ch == '\t' {
                cursor_x += glyph_size[0] * 4.0;
                continue;
            }
            if ch == ' ' {
                cursor_x += glyph_size[0];
                continue;
            }
            sprites.push(SpriteInstance {
                sprite_id: *sprite_id_seed,
                kind: SpriteKind::Generic,
                position: [cursor_x, y, z],
                size: glyph_size,
                rotation_rad: 0.0,
                color_rgba: color,
                texture_slot: Self::console_glyph_slot(ch),
                layer,
            });
            *sprite_id_seed = sprite_id_seed.saturating_add(1);
            cursor_x += glyph_size[0] * 0.86;
        }
    }

    pub(super) fn push_console_rect(
        sprites: &mut Vec<SpriteInstance>,
        sprite_id_seed: &mut u64,
        pos: [f32; 3],
        size: [f32; 2],
        color: [f32; 4],
        layer: i16,
    ) {
        sprites.push(SpriteInstance {
            sprite_id: *sprite_id_seed,
            kind: SpriteKind::Hud,
            position: pos,
            size,
            rotation_rad: 0.0,
            color_rgba: color,
            texture_slot: 1,
            layer,
        });
        *sprite_id_seed = sprite_id_seed.saturating_add(1);
    }

    pub(super) fn append_console_overlay_sprites(
        console: &RuntimeConsoleState,
        layout: ConsoleUiLayout,
        sprites: &mut Vec<SpriteInstance>,
    ) {
        if !console.open {
            return;
        }
        let mut sprite_id_seed = 9_200_000u64 + sprites.len() as u64;

        // Full-screen semi-transparent console shell.
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            [0.0, 0.0, 0.99],
            [2.0, 2.0],
            [0.02, 0.08, 0.03, 0.80],
            29_000,
        );
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            layout.pos(0.0, 0.80, 0.98),
            layout.rect_size(1.92, 0.26),
            [0.06, 0.20, 0.09, 0.92],
            29_001,
        );
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            layout.pos(-0.50, 0.37, 0.98),
            layout.rect_size(0.92, 0.76),
            [0.04, 0.12, 0.05, 0.92],
            29_001,
        );
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            layout.pos(0.52, 0.37, 0.98),
            layout.rect_size(0.90, 0.76),
            [0.04, 0.12, 0.05, 0.92],
            29_001,
        );
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            layout.pos(0.0, -0.72, 0.98),
            layout.rect_size(1.92, 0.44),
            [0.03, 0.10, 0.04, 0.94],
            29_001,
        );
        // Input/editable box.
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            [layout.command_center.0, layout.command_center.1, 0.97],
            [layout.command_size.0, layout.command_size.1],
            [0.02, 0.18, 0.05, 0.96],
            29_002,
        );
        // Send button (right side of command input).
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            [layout.send_center.0, layout.send_center.1, 0.97],
            [layout.send_size.0, layout.send_size.1],
            [0.04, 0.28, 0.08, 0.98],
            29_003,
        );

        let header_color = [0.74, 1.0, 0.72, 1.0];
        let info_color = [0.62, 0.96, 0.62, 1.0];
        let mut y = 0.90;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "TLAPP CLI OVERLAY",
            -0.93 * layout.sx,
            y * layout.sy,
            0.96,
            layout.glyph_size(0.028, 0.046),
            header_color,
            29_010,
        );
        y -= 0.065;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "GREEN TEXT | FULLSCREEN SEMI-TRANSPARENT | ERRORS BLINK RED",
            -0.93 * layout.sx,
            y * layout.sy,
            0.96,
            layout.glyph_size(0.018, 0.032),
            info_color,
            29_010,
        );
        y -= 0.055;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "HOTKEYS: CTRL+F1/F1 TOGGLE | ENTER RUN | ARROW UP/DOWN HISTORY",
            -0.93 * layout.sx,
            y * layout.sy,
            0.96,
            layout.glyph_size(0.018, 0.032),
            info_color,
            29_010,
        );
        y -= 0.055;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "LOG SCROLL: WHEEL / PGUP/PGDN | HOME/END",
            -0.93 * layout.sx,
            y * layout.sy,
            0.96,
            layout.glyph_size(0.016, 0.029),
            info_color,
            29_010,
        );
        y -= 0.055;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            &format!(
                "TAB SWITCH BOX | ACTIVE: {}",
                console.edit_target.label().to_uppercase()
            ),
            -0.93 * layout.sx,
            y * layout.sy,
            0.96,
            layout.glyph_size(0.018, 0.032),
            [0.84, 0.98, 0.74, 1.0],
            29_010,
        );

        // Command list box.
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "COMMAND LIST",
            -0.93 * layout.sx,
            0.63 * layout.sy,
            0.96,
            layout.glyph_size(0.02, 0.036),
            header_color,
            29_010,
        );
        let mut cmd_y = 0.58;
        for cmd in CONSOLE_HELP_COMMANDS.iter().take(14) {
            Self::push_console_text_line(
                sprites,
                &mut sprite_id_seed,
                cmd,
                -0.93 * layout.sx,
                cmd_y * layout.sy,
                0.96,
                layout.glyph_size(0.016, 0.029),
                [0.58, 0.92, 0.58, 1.0],
                29_010,
            );
            cmd_y -= 0.044;
        }

        // Quick settings / editable values.
        let quick_left = 0.08;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "EDIT BOXES",
            quick_left * layout.sx,
            0.63 * layout.sy,
            0.96,
            layout.glyph_size(0.02, 0.036),
            header_color,
            29_010,
        );
        let settings = [
            format!("FPS CAP: {}", console.quick_fps_cap),
            format!("RENDER DISTANCE: {}", console.quick_render_distance),
            format!("FSR SHARPNESS: {}", console.quick_fsr_sharpness),
            format!("MSAA: {}", console.quick_msaa),
        ];
        let active_fps = console.edit_target == RuntimeConsoleEditTarget::FpsCap;
        let active_rd = console.edit_target == RuntimeConsoleEditTarget::RenderDistance;
        let active_sharp = console.edit_target == RuntimeConsoleEditTarget::FsrSharpness;
        let active_msaa = console.edit_target == RuntimeConsoleEditTarget::Msaa;
        let mut settings_y = 0.57;
        for (index, line) in settings.into_iter().enumerate() {
            let is_active = match index {
                0 => active_fps,
                1 => active_rd,
                2 => active_sharp,
                _ => active_msaa,
            };
            Self::push_console_text_line(
                sprites,
                &mut sprite_id_seed,
                &if is_active { format!("> {line}") } else { line },
                quick_left * layout.sx,
                settings_y * layout.sy,
                0.96,
                layout.glyph_size(0.018, 0.032),
                if is_active {
                    [0.90, 1.00, 0.82, 1.0]
                } else {
                    [0.64, 0.98, 0.64, 1.0]
                },
                29_010,
            );
            settings_y -= 0.055;
        }
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "BOX INPUT: TYPE VALUE + COMMAND (GFX.FPS_CAP / GFX.RENDER_DISTANCE ...)",
            quick_left * layout.sx,
            0.39 * layout.sy,
            0.96,
            layout.glyph_size(0.014, 0.026),
            [0.52, 0.86, 0.52, 1.0],
            29_010,
        );
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            [layout.apply_center.0, layout.apply_center.1, 0.97],
            [layout.apply_size.0, layout.apply_size.1],
            [0.03, 0.22, 0.06, 0.96],
            29_002,
        );
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "[CLICK] APPLY ALL BOXES",
            0.18 * layout.sx,
            0.32 * layout.sy,
            0.96,
            layout.glyph_size(0.016, 0.028),
            [0.84, 1.0, 0.84, 1.0],
            29_011,
        );
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "SEND",
            layout.send_center.0 - layout.send_size.0 * 0.22,
            layout.send_center.1,
            0.96,
            layout.glyph_size(0.020, 0.033),
            [0.90, 1.0, 0.88, 1.0],
            29_021,
        );

        // Live input line.
        let glyph = layout.glyph_size(0.022, 0.038);
        let max_chars = ((layout.command_size.0 * 0.90) / (glyph[0] * 0.86))
            .floor()
            .max(8.0) as usize;
        let mut input_tail = console.input_line.clone();
        let input_len = input_tail.chars().count();
        if input_len > max_chars.saturating_sub(2) {
            let keep = max_chars.saturating_sub(3);
            input_tail = input_tail
                .chars()
                .rev()
                .take(keep)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>();
            input_tail = format!("...{input_tail}");
        }
        let input_line = format!("> {}", input_tail);
        let input_active = console.edit_target == RuntimeConsoleEditTarget::Command;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            &input_line,
            layout.command_center.0 - layout.command_size.0 * 0.48,
            layout.command_center.1,
            0.96,
            glyph,
            if input_active {
                [0.90, 1.0, 0.90, 1.0]
            } else {
                [0.66, 0.90, 0.66, 1.0]
            },
            29_020,
        );

        // Output list with blinking red errors.
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "OUTPUT",
            -0.93 * layout.sx,
            -0.66 * layout.sy,
            0.96,
            layout.glyph_size(0.02, 0.036),
            header_color,
            29_010,
        );
        let now = Instant::now();
        let line_step = (0.047 * layout.text_scale).max(0.026);
        let out_top = -0.72f32;
        let out_bottom = -0.90f32;
        let visible_logs = (((out_top - out_bottom) / line_step).floor() as usize).max(3);
        let filtered_logs = console
            .log_lines
            .iter()
            .filter(|line| console.log_filter.matches(line.level))
            .collect::<Vec<_>>();
        let total_logs = filtered_logs.len();
        let tail_start = if let Some(limit) = console.log_tail_limit {
            total_logs.saturating_sub(limit)
        } else {
            0
        };
        let end = total_logs.saturating_sub(console.log_scroll);
        let start = end.saturating_sub(visible_logs).max(tail_start);
        let mut out_y = out_top;
        for line in filtered_logs
            .iter()
            .skip(start)
            .take(end.saturating_sub(start))
        {
            let age_ms = now.saturating_duration_since(line.timestamp).as_millis();
            let color = match line.level {
                RuntimeConsoleLogLevel::Info => [0.70, 0.98, 0.70, 1.0],
                RuntimeConsoleLogLevel::Error => {
                    if ((age_ms / CONSOLE_ERROR_BLINK_MS) % 2) == 0 {
                        [1.0, 0.32, 0.32, 1.0]
                    } else {
                        [0.68, 0.12, 0.12, 1.0]
                    }
                }
            };
            let text = format!("[{:>5}ms] {}", age_ms.min(99_999), line.message);
            Self::push_console_text_line(
                sprites,
                &mut sprite_id_seed,
                &text,
                -0.93 * layout.sx,
                out_y * layout.sy,
                0.96,
                layout.glyph_size(0.016, 0.028),
                color,
                29_010,
            );
            out_y -= line_step;
            if out_y < out_bottom {
                break;
            }
        }
        let filter_label = match console.log_filter {
            RuntimeConsoleLogFilter::All => "all",
            RuntimeConsoleLogFilter::Info => "info",
            RuntimeConsoleLogFilter::Error => "error",
        };
        let tail_label = console
            .log_tail_limit
            .map(|n| n.to_string())
            .unwrap_or_else(|| "off".to_string());
        if console.log_scroll > 0
            || !matches!(console.log_filter, RuntimeConsoleLogFilter::All)
            || console.log_tail_limit.is_some()
        {
            Self::push_console_text_line(
                sprites,
                &mut sprite_id_seed,
                &format!(
                    "SCROLL {} | FILTER {} | TAIL {}",
                    console.log_scroll, filter_label, tail_label
                ),
                0.72 * layout.sx,
                -0.66 * layout.sy,
                0.96,
                layout.glyph_size(0.013, 0.024),
                [0.74, 0.96, 0.74, 1.0],
                29_010,
            );
        }
    }

    pub(super) fn point_in_rect(point: (f32, f32), center: (f32, f32), size: (f32, f32)) -> bool {
        let (px, py) = point;
        let (cx, cy) = center;
        let (sx, sy) = size;
        (px >= cx - sx * 0.5)
            && (px <= cx + sx * 0.5)
            && (py >= cy - sy * 0.5)
            && (py <= cy + sy * 0.5)
    }

    pub(super) fn apply_all_console_quick_boxes(&mut self) {
        let previous_target = self.console.edit_target;
        self.console.edit_target = RuntimeConsoleEditTarget::FpsCap;
        let _ = self.apply_active_console_edit_target();
        self.console.edit_target = RuntimeConsoleEditTarget::RenderDistance;
        let _ = self.apply_active_console_edit_target();
        self.console.edit_target = RuntimeConsoleEditTarget::FsrSharpness;
        let _ = self.apply_active_console_edit_target();
        self.console.edit_target = RuntimeConsoleEditTarget::Msaa;
        let _ = self.apply_active_console_edit_target();
        self.console.edit_target = previous_target;
    }

    pub(super) fn handle_console_left_click(&mut self) {
        let Some(cursor) = self.cursor_ndc() else {
            return;
        };
        let layout = ConsoleUiLayout::from_size(self.size);
        if Self::point_in_rect(cursor, layout.send_center, layout.send_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::Command;
            let _ = self.submit_console_command(self.console.input_line.clone());
            return;
        }
        if Self::point_in_rect(cursor, layout.command_center, layout.command_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::Command;
            self.console_feedback("active edit box: command");
            return;
        }
        if Self::point_in_rect(cursor, layout.fps_center, layout.fps_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::FpsCap;
            self.console_feedback("active edit box: fps_cap");
            return;
        }
        if Self::point_in_rect(cursor, layout.distance_center, layout.distance_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::RenderDistance;
            self.console_feedback("active edit box: render_distance");
            return;
        }
        if Self::point_in_rect(cursor, layout.sharpness_center, layout.sharpness_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::FsrSharpness;
            self.console_feedback("active edit box: fsr_sharpness");
            return;
        }
        if Self::point_in_rect(cursor, layout.msaa_center, layout.msaa_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::Msaa;
            self.console_feedback("active edit box: msaa");
            return;
        }
        if Self::point_in_rect(cursor, layout.apply_center, layout.apply_size) {
            self.apply_all_console_quick_boxes();
        }
    }
}
