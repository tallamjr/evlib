use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use warp::Filter;
use warp::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::ev_core::Event;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebServerConfig {
    pub port: u16,
    pub host: String,
    pub max_clients: usize,
    pub event_batch_size: usize,
    pub batch_interval_ms: u64,
}

impl Default for WebServerConfig {
    fn default() -> Self {
        Self {
            port: 3030,
            host: "127.0.0.1".to_string(),
            max_clients: 100,
            event_batch_size: 1000,
            batch_interval_ms: 16, // ~60fps
        }
    }
}

#[derive(Debug)]
pub struct WebSocketClient {
    pub id: Uuid,
    pub tx: tokio::sync::mpsc::UnboundedSender<Message>,
}

pub struct EventBroadcaster {
    clients: HashMap<Uuid, WebSocketClient>,
    event_buffer: Vec<Event>,
}

impl EventBroadcaster {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            event_buffer: Vec::with_capacity(10000),
        }
    }

    pub fn add_client(&mut self, client: WebSocketClient) -> Result<(), String> {
        self.clients.insert(client.id, client);
        Ok(())
    }

    pub fn remove_client(&mut self, id: &Uuid) {
        self.clients.remove(id);
    }

    pub async fn broadcast_events(&mut self, events: Vec<Event>) {
        let event_count = events.len();
        self.event_buffer.extend(events);
        
        // More aggressive batching for higher throughput - send smaller, more frequent batches
        if self.event_buffer.len() >= 100 || (!self.clients.is_empty() && self.event_buffer.len() >= 20) {
            let batch = std::mem::replace(&mut self.event_buffer, Vec::with_capacity(5000));
            if !batch.is_empty() {
                let message = Self::serialize_events(&batch);
                if batch.len() > 50 { // Only log larger batches to reduce spam
                    println!("Broadcasting {} events ({} bytes) to {} clients", 
                        batch.len(), message.len(), self.clients.len());
                }
                
                let disconnected_clients: Vec<Uuid> = self.clients
                    .iter()
                    .filter_map(|(id, client)| {
                        match client.tx.send(Message::binary(message.clone())) {
                            Ok(_) => {
                                println!("  ✓ Sent to client {}", id);
                                None
                            }
                            Err(e) => {
                                println!("  ✗ Failed to send to client {}: {}", id, e);
                                Some(*id)
                            }
                        }
                    })
                    .collect();

                for id in disconnected_clients {
                    self.remove_client(&id);
                }
            }
        } else {
            // Only log buffering for significant accumulations
            if self.event_buffer.len() % 100 == 0 && self.event_buffer.len() > 0 {
                println!("Buffering {} events (total: {}, clients: {})", 
                    event_count, self.event_buffer.len(), self.clients.len());
            }
        }
    }

    fn serialize_events(events: &[Event]) -> Vec<u8> {
        // Simple binary format for now
        // Header: message_type (1) + timestamp (8) + event_count (4) = 13 bytes
        // Events: x (2) + y (2) + timestamp (8) + polarity (1) = 13 bytes per event
        let mut buffer = Vec::with_capacity(13 + events.len() * 13);
        
        // Header
        buffer.push(1u8); // Message type: events
        if let Some(first_event) = events.first() {
            // Convert f64 timestamp to u64 microseconds
            let timestamp_us = (first_event.t * 1_000_000.0) as u64;
            buffer.extend_from_slice(&timestamp_us.to_le_bytes());
        } else {
            buffer.extend_from_slice(&0u64.to_le_bytes());
        }
        buffer.extend_from_slice(&(events.len() as u32).to_le_bytes());

        // Events
        for event in events {
            buffer.extend_from_slice(&event.x.to_le_bytes());
            buffer.extend_from_slice(&event.y.to_le_bytes());
            // Convert f64 timestamp to u64 microseconds
            let timestamp_us = (event.t * 1_000_000.0) as u64;
            buffer.extend_from_slice(&timestamp_us.to_le_bytes());
            let polarity = if event.polarity > 0 { 1u8 } else { 0u8 };
            buffer.push(polarity);
        }

        buffer
    }
}

pub struct EventWebServer {
    config: WebServerConfig,
    broadcaster: Arc<Mutex<EventBroadcaster>>,
}

impl EventWebServer {
    pub fn new(config: WebServerConfig) -> Self {
        Self {
            config,
            broadcaster: Arc::new(Mutex::new(EventBroadcaster::new())),
        }
    }

    pub async fn run(&self) {
        let broadcaster = self.broadcaster.clone();
        let websocket_route = warp::path("ws")
            .and(warp::ws())
            .and(warp::any().map(move || broadcaster.clone()))
            .map(|ws: warp::ws::Ws, broadcaster: Arc<Mutex<EventBroadcaster>>| {
                ws.on_upgrade(move |socket| handle_websocket(socket, broadcaster))
            });

        let static_route = warp::path::end()
            .and(warp::get())
            .map(|| {
                warp::reply::html(include_str!("../../static/index.html"))
            });

        let routes = websocket_route.or(static_route);

        println!("WebSocket server listening on {}:{}", self.config.host, self.config.port);
        warp::serve(routes)
            .run(([127, 0, 0, 1], self.config.port))
            .await;
    }

    pub fn broadcaster(&self) -> Arc<Mutex<EventBroadcaster>> {
        self.broadcaster.clone()
    }
}

async fn handle_websocket(ws: WebSocket, broadcaster: Arc<Mutex<EventBroadcaster>>) {
    let client_id = Uuid::new_v4();
    let (mut ws_tx, mut ws_rx) = ws.split();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

    let client = WebSocketClient {
        id: client_id,
        tx: tx.clone(),
    };

    broadcaster.lock().await.add_client(client).unwrap();
    println!("Client {} connected (total clients: {})", 
        client_id, broadcaster.lock().await.clients.len());

    // Spawn task to forward messages from channel to websocket
    let mut send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if ws_tx.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Handle incoming messages (if any)
    let mut recv_task = tokio::spawn(async move {
        while let Some(result) = ws_rx.next().await {
            match result {
                Ok(msg) => {
                    // Handle control messages if needed
                    if msg.is_close() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }

    broadcaster.lock().await.remove_client(&client_id);
    println!("Client {} disconnected (remaining clients: {})", 
        client_id, broadcaster.lock().await.clients.len());
}