//! Terminal-based event visualization using Ratatui
//!
//! This module provides ultra-high-performance event visualization directly in the terminal
//! using Ratatui. This eliminates GUI overhead and provides the fastest possible visualization.

use crate::ev_core::Event;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{
        canvas::{Canvas, Points},
        Block, Borders, Clear, List, ListItem, Paragraph,
    },
    Frame, Terminal,
};
use std::{
    collections::VecDeque,
    io::{self, Stdout},
    time::{Duration, Instant},
};

/// Configuration for terminal-based event visualization
#[derive(Debug, Clone)]
pub struct TerminalVisualizationConfig {
    /// Event decay time in milliseconds
    pub event_decay_ms: f32,
    /// Maximum events to display
    pub max_events: usize,
    /// Update rate in Hz
    pub target_fps: f32,
    /// Show detailed statistics
    pub show_stats: bool,
    /// Canvas resolution scale factor
    pub canvas_scale: f32,
}

impl Default for TerminalVisualizationConfig {
    fn default() -> Self {
        Self {
            event_decay_ms: 100.0,
            max_events: 1000,
            target_fps: 60.0,
            show_stats: true,
            canvas_scale: 1.0,
        }
    }
}

/// Statistics for terminal visualization
#[derive(Debug, Clone, Default)]
pub struct TerminalVisualizationStats {
    pub frames_rendered: u64,
    pub events_processed: u64,
    pub current_fps: f32,
    pub avg_events_per_frame: f32,
    pub terminal_size: (u16, u16),
    pub canvas_size: (u16, u16),
}

/// Terminal-based event visualizer using Ratatui
pub struct TerminalEventVisualizer {
    config: TerminalVisualizationConfig,
    terminal: Terminal<CrosstermBackend<Stdout>>,

    // Event management
    event_buffer: VecDeque<(Event, Instant)>,

    // Performance tracking
    stats: TerminalVisualizationStats,
    last_frame_time: Instant,
    fps_history: VecDeque<f32>,

    // UI state
    should_quit: bool,
    paused: bool,
    show_help: bool,
    canvas_bounds: (f64, f64, f64, f64), // (x_min, y_min, x_max, y_max)
}

impl TerminalEventVisualizer {
    /// Create a new terminal visualizer
    pub fn new(config: TerminalVisualizationConfig) -> io::Result<Self> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        let mut visualizer = Self {
            config,
            terminal,
            event_buffer: VecDeque::with_capacity(10000),
            stats: TerminalVisualizationStats::default(),
            last_frame_time: Instant::now(),
            fps_history: VecDeque::with_capacity(60),
            should_quit: false,
            paused: false,
            show_help: false,
            canvas_bounds: (0.0, 0.0, 640.0, 480.0),
        };

        // Get initial terminal size
        let size = visualizer.terminal.size()?;
        visualizer.stats.terminal_size = (size.width, size.height);
        visualizer.stats.canvas_size =
            (size.width.saturating_sub(4), size.height.saturating_sub(8));

        Ok(visualizer)
    }

    /// Add events to the visualization buffer
    pub fn add_events(&mut self, events: Vec<Event>) {
        let now = Instant::now();

        // Update canvas bounds based on events
        for event in &events {
            let x = event.x as f64;
            let y = event.y as f64;

            if x < self.canvas_bounds.0 {
                self.canvas_bounds.0 = x;
            }
            if y < self.canvas_bounds.1 {
                self.canvas_bounds.1 = y;
            }
            if x > self.canvas_bounds.2 {
                self.canvas_bounds.2 = x;
            }
            if y > self.canvas_bounds.3 {
                self.canvas_bounds.3 = y;
            }
        }

        // Add events to buffer
        self.stats.events_processed += events.len() as u64;
        for event in events {
            if self.event_buffer.len() >= self.config.max_events {
                self.event_buffer.pop_front();
            }
            self.event_buffer.push_back((event, now));
        }

        // Remove old events
        let decay_duration = Duration::from_millis(self.config.event_decay_ms as u64);
        while let Some((_, timestamp)) = self.event_buffer.front() {
            if now.duration_since(*timestamp) > decay_duration {
                self.event_buffer.pop_front();
            } else {
                break;
            }
        }
    }

    /// Handle keyboard input
    pub fn handle_input(&mut self) -> io::Result<()> {
        if event::poll(Duration::from_millis(0))? {
            if let CrosstermEvent::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
                    KeyCode::Char('p') | KeyCode::Char(' ') => self.paused = !self.paused,
                    KeyCode::Char('h') | KeyCode::F(1) => self.show_help = !self.show_help,
                    KeyCode::Char('r') => {
                        self.event_buffer.clear();
                        self.stats = TerminalVisualizationStats::default();
                        self.fps_history.clear();
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        self.config.event_decay_ms =
                            (self.config.event_decay_ms + 10.0).min(1000.0);
                    }
                    KeyCode::Char('-') => {
                        self.config.event_decay_ms = (self.config.event_decay_ms - 10.0).max(10.0);
                    }
                    KeyCode::Char('s') => {
                        self.config.show_stats = !self.config.show_stats;
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    /// Render a single frame
    pub fn render_frame(&mut self) -> io::Result<()> {
        // Update FPS
        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        if frame_time > 0.0 {
            let current_fps = 1.0 / frame_time;
            self.fps_history.push_back(current_fps);
            if self.fps_history.len() > 60 {
                self.fps_history.pop_front();
            }
            self.stats.current_fps =
                self.fps_history.iter().sum::<f32>() / self.fps_history.len() as f32;
        }

        self.stats.frames_rendered += 1;
        if self.stats.frames_rendered > 0 {
            self.stats.avg_events_per_frame =
                self.stats.events_processed as f32 / self.stats.frames_rendered as f32;
        }

        // Extract needed data for rendering to avoid borrowing issues
        let config = self.config.clone();
        let stats = self.stats.clone();
        let canvas_bounds = self.canvas_bounds;
        let event_buffer = self.event_buffer.clone();
        let _should_quit = self.should_quit;
        let paused = self.paused;
        let show_help = self.show_help;

        self.terminal.draw(move |f| {
            render_ui_static(
                f,
                &config,
                &stats,
                canvas_bounds,
                &event_buffer,
                paused,
                show_help,
            )
        })?;
        Ok(())
    }

    /// Check if should quit
    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    /// Check if paused
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &TerminalVisualizationStats {
        &self.stats
    }

    /// Main event loop
    pub fn run_event_loop<F>(&mut self, mut event_source: F) -> io::Result<()>
    where
        F: FnMut() -> Vec<Event>,
    {
        let target_frame_time = Duration::from_secs_f32(1.0 / self.config.target_fps);

        while !self.should_quit() {
            let loop_start = Instant::now();

            // Handle input
            self.handle_input()?;

            // Get new events if not paused
            if !self.is_paused() {
                let events = event_source();
                self.add_events(events);
            }

            // Render frame
            self.render_frame()?;

            // Frame rate limiting
            let loop_time = loop_start.elapsed();
            if loop_time < target_frame_time {
                std::thread::sleep(target_frame_time - loop_time);
            }
        }

        Ok(())
    }
}

impl Drop for TerminalEventVisualizer {
    fn drop(&mut self) {
        // Restore terminal
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        let _ = self.terminal.show_cursor();
    }
}

/// Static render function to avoid borrowing issues
fn render_ui_static(
    f: &mut Frame,
    config: &TerminalVisualizationConfig,
    stats: &TerminalVisualizationStats,
    canvas_bounds: (f64, f64, f64, f64),
    event_buffer: &VecDeque<(Event, Instant)>,
    paused: bool,
    show_help: bool,
) {
    let size = f.area();

    if show_help {
        render_help_static(f);
        return;
    }

    // Create layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(3),                                     // Header
            Constraint::Min(10),                                       // Canvas
            Constraint::Length(if config.show_stats { 8 } else { 3 }), // Stats/Footer
        ])
        .split(size);

    // Render header
    render_header_static(f, chunks[0], stats, paused);

    // Render event canvas
    render_canvas_static(f, chunks[1], config, canvas_bounds, event_buffer);

    // Render stats/footer
    if config.show_stats {
        render_stats_static(f, chunks[2], stats, config);
    } else {
        render_footer_static(f, chunks[2]);
    }
}

fn render_header_static(
    f: &mut Frame,
    area: Rect,
    stats: &TerminalVisualizationStats,
    paused: bool,
) {
    let status = if paused { "PAUSED" } else { "RUNNING" };
    let status_style = if paused {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::Green)
    };

    let header = Paragraph::new(vec![Line::from(vec![
        Span::styled("Event Stream Visualizer ", Style::default().fg(Color::Cyan)),
        Span::styled("(Terminal Mode)", Style::default().fg(Color::Gray)),
        Span::raw(" | "),
        Span::styled(status, status_style),
        Span::raw(" | "),
        Span::styled(
            format!("{:.1} FPS", stats.current_fps),
            Style::default().fg(Color::Yellow),
        ),
    ])])
    .block(Block::default().borders(Borders::ALL))
    .alignment(Alignment::Center);

    f.render_widget(header, area);
}

fn render_canvas_static(
    f: &mut Frame,
    area: Rect,
    config: &TerminalVisualizationConfig,
    canvas_bounds: (f64, f64, f64, f64),
    event_buffer: &VecDeque<(Event, Instant)>,
) {
    let canvas = Canvas::default()
        .block(Block::default().borders(Borders::ALL).title("Events"))
        .x_bounds([canvas_bounds.0, canvas_bounds.2])
        .y_bounds([canvas_bounds.1, canvas_bounds.3])
        .paint(|ctx| {
            // Separate positive and negative events
            let now = Instant::now();
            let decay_ms = config.event_decay_ms;

            let mut positive_events = Vec::new();
            let mut negative_events = Vec::new();

            for (event, timestamp) in event_buffer {
                let age_ms = now.duration_since(*timestamp).as_millis() as f32;
                if age_ms < decay_ms {
                    // Fix coordinate transformation:
                    // 1. Flip X-axis: subtract from max to reverse horizontal movement
                    // 2. Flip Y-axis: subtract from max to fix upside-down rendering
                    let x = canvas_bounds.2 - event.x as f64;
                    let y = canvas_bounds.3 - event.y as f64;

                    if event.polarity > 0 {
                        positive_events.push((x, y));
                    } else {
                        negative_events.push((x, y));
                    }
                }
            }

            // Draw positive events (red)
            if !positive_events.is_empty() {
                ctx.draw(&Points {
                    coords: &positive_events,
                    color: Color::Red,
                });
            }

            // Draw negative events (blue)
            if !negative_events.is_empty() {
                ctx.draw(&Points {
                    coords: &negative_events,
                    color: Color::Blue,
                });
            }
        });

    f.render_widget(canvas, area);
}

fn render_stats_static(
    f: &mut Frame,
    area: Rect,
    stats: &TerminalVisualizationStats,
    config: &TerminalVisualizationConfig,
) {
    let stats_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Left column - Performance stats
    let perf_items = vec![
        ListItem::new(format!("FPS: {:.1}", stats.current_fps)),
        ListItem::new(format!("Frames: {}", stats.frames_rendered)),
        ListItem::new(format!("Events: {}", stats.events_processed)),
        ListItem::new(format!("Events/Frame: {:.1}", stats.avg_events_per_frame)),
    ];

    let perf_list =
        List::new(perf_items).block(Block::default().borders(Borders::ALL).title("Performance"));

    f.render_widget(perf_list, stats_chunks[0]);

    // Right column - Configuration
    let config_items = vec![
        ListItem::new(format!("Decay: {:.0}ms", config.event_decay_ms)),
        ListItem::new(format!("Max Events: {}", config.max_events)),
        ListItem::new(format!(
            "Terminal: {}x{}",
            stats.terminal_size.0, stats.terminal_size.1
        )),
        ListItem::new(format!(
            "Canvas: {}x{}",
            stats.canvas_size.0, stats.canvas_size.1
        )),
    ];

    let config_list = List::new(config_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Configuration"),
    );

    f.render_widget(config_list, stats_chunks[1]);
}

fn render_footer_static(f: &mut Frame, area: Rect) {
    let footer = Paragraph::new("Press 'h' for help, 'q' to quit, 'p' to pause, 'r' to reset")
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);

    f.render_widget(footer, area);
}

fn render_help_static(f: &mut Frame) {
    let size = f.area();
    let area = Rect {
        x: size.width / 4,
        y: size.height / 4,
        width: size.width / 2,
        height: size.height / 2,
    };

    f.render_widget(Clear, area);

    let help_text = vec![
        Line::from("Terminal Event Visualizer Help"),
        Line::from(""),
        Line::from("Controls:"),
        Line::from("  q, Esc    - Quit"),
        Line::from("  p, Space  - Pause/Resume"),
        Line::from("  r         - Reset statistics"),
        Line::from("  s         - Toggle statistics"),
        Line::from("  +/-       - Adjust event decay time"),
        Line::from("  h, F1     - Toggle this help"),
        Line::from(""),
        Line::from("Events:"),
        Line::from("  Red dots  - Positive events"),
        Line::from("  Blue dots - Negative events"),
        Line::from(""),
        Line::from("Press any key to close help"),
    ];

    let help = Paragraph::new(help_text)
        .block(Block::default().borders(Borders::ALL).title("Help"))
        .alignment(Alignment::Left);

    f.render_widget(help, area);
}

/// Create a simple terminal event stream viewer
pub fn create_terminal_event_viewer(
    config: TerminalVisualizationConfig,
) -> io::Result<TerminalEventVisualizer> {
    TerminalEventVisualizer::new(config)
}
