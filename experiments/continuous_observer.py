#!/usr/bin/env python3
"""
Continuous Background Observation Pipeline

Run 24/7 observation for 60+ days to accumulate 4000+ events.

Features:
- SQLite database (not RAM-limited)
- Multiple parallel streams
- Event detection and clustering
- Routine extraction
- Anomaly flagging

Architecture:
- Main process: Coordination
- Worker processes: Stream observation
- Database: Persistent storage
"""

import sys
sys.path.insert(0, '..')
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import sqlite3
import time
import signal
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import pickle
import multiprocessing as mp

import torch
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class StreamConfig:
    """Configuration for an observation stream."""
    stream_id: str
    source_type: str  # 'webcam', 'file', 'synthetic'
    source_url: str
    sample_interval: float = 1.0  # seconds between samples
    description: str = ""


class ObservationDatabase:
    """
    SQLite database for persistent event storage.
    
    Supports millions of events without RAM issues.
    """
    
    def __init__(self, db_path: str = 'observations.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Events table
        c.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stream_id TEXT,
                timestamp TEXT,
                event_type TEXT,
                change_magnitude REAL,
                activation BLOB,
                cluster_id INTEGER,
                is_anomaly INTEGER DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        # Clusters table
        c.execute('''
            CREATE TABLE IF NOT EXISTS clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prototype BLOB,
                event_count INTEGER DEFAULT 0,
                label TEXT,
                domain TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Routines table
        c.execute('''
            CREATE TABLE IF NOT EXISTS routines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT,
                frequency INTEGER,
                time_of_day TEXT,
                day_of_week TEXT,
                stream_id TEXT,
                confidence REAL
            )
        ''')
        
        # Stream stats table
        c.execute('''
            CREATE TABLE IF NOT EXISTS stream_stats (
                stream_id TEXT PRIMARY KEY,
                frames_processed INTEGER DEFAULT 0,
                events_detected INTEGER DEFAULT 0,
                last_update TEXT,
                status TEXT
            )
        ''')
        
        # Indices for fast queries
        c.execute('CREATE INDEX IF NOT EXISTS idx_events_stream ON events(stream_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_events_cluster ON events(cluster_id)')
        
        conn.commit()
        conn.close()
    
    def add_event(
        self,
        stream_id: str,
        event_type: str,
        change_magnitude: float,
        activation: Optional[torch.Tensor] = None,
        cluster_id: Optional[int] = None,
        is_anomaly: bool = False,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Add event to database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        activation_blob = None
        if activation is not None:
            activation_blob = pickle.dumps(activation.cpu().numpy())
        
        c.execute('''
            INSERT INTO events 
            (stream_id, timestamp, event_type, change_magnitude, activation, cluster_id, is_anomaly, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stream_id,
            datetime.now().isoformat(),
            event_type,
            change_magnitude,
            activation_blob,
            cluster_id,
            1 if is_anomaly else 0,
            json.dumps(metadata or {}),
        ))
        
        event_id = c.lastrowid
        
        conn.commit()
        conn.close()
        
        return event_id
    
    def add_cluster(
        self,
        prototype: torch.Tensor,
        label: str = "",
        domain: str = "general",
    ) -> int:
        """Add cluster to database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        prototype_blob = pickle.dumps(prototype.cpu().numpy())
        now = datetime.now().isoformat()
        
        c.execute('''
            INSERT INTO clusters (prototype, event_count, label, domain, created_at, updated_at)
            VALUES (?, 0, ?, ?, ?, ?)
        ''', (prototype_blob, label, domain, now, now))
        
        cluster_id = c.lastrowid
        
        conn.commit()
        conn.close()
        
        return cluster_id
    
    def update_stream_stats(
        self,
        stream_id: str,
        frames: int = 0,
        events: int = 0,
        status: str = "running",
    ):
        """Update stream statistics."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO stream_stats (stream_id, frames_processed, events_detected, last_update, status)
            VALUES (
                ?,
                COALESCE((SELECT frames_processed FROM stream_stats WHERE stream_id = ?), 0) + ?,
                COALESCE((SELECT events_detected FROM stream_stats WHERE stream_id = ?), 0) + ?,
                ?,
                ?
            )
        ''', (stream_id, stream_id, frames, stream_id, events, datetime.now().isoformat(), status))
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> Dict:
        """Get overall statistics."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM events')
        total_events = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM clusters')
        total_clusters = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM events WHERE is_anomaly = 1')
        total_anomalies = c.fetchone()[0]
        
        c.execute('SELECT stream_id, frames_processed, events_detected, status FROM stream_stats')
        streams = {row[0]: {'frames': row[1], 'events': row[2], 'status': row[3]} for row in c.fetchall()}
        
        c.execute('SELECT event_type, COUNT(*) FROM events GROUP BY event_type')
        event_types = {row[0]: row[1] for row in c.fetchall()}
        
        conn.close()
        
        return {
            'total_events': total_events,
            'total_clusters': total_clusters,
            'total_anomalies': total_anomalies,
            'streams': streams,
            'event_types': event_types,
        }
    
    def extract_routines(self) -> List[Dict]:
        """Extract recurring patterns from events."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Find patterns by hour
        c.execute('''
            SELECT 
                stream_id,
                event_type,
                strftime('%H', timestamp) as hour,
                COUNT(*) as frequency
            FROM events
            GROUP BY stream_id, event_type, hour
            HAVING frequency > 5
            ORDER BY frequency DESC
        ''')
        
        routines = []
        for row in c.fetchall():
            routine = {
                'stream_id': row[0],
                'event_type': row[1],
                'hour': row[2],
                'frequency': row[3],
            }
            routines.append(routine)
            
            # Store in routines table
            c.execute('''
                INSERT INTO routines (pattern, frequency, time_of_day, stream_id, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (row[1], row[3], f"{row[2]}:00", row[0], min(row[3] / 100.0, 1.0)))
        
        conn.commit()
        conn.close()
        
        return routines


class StreamObserver:
    """
    Observe a single stream and detect events.
    
    Runs in a separate process.
    """
    
    def __init__(
        self,
        config: StreamConfig,
        db_path: str,
        brain_path: Optional[str] = None,
    ):
        self.config = config
        self.db = ObservationDatabase(db_path)
        self.brain = None
        self.brain_path = brain_path
        
        self.running = True
        self.frames_processed = 0
        self.events_detected = 0
    
    def load_brain(self):
        """Load brain model (if available)."""
        if self.brain_path and Path(self.brain_path).exists():
            try:
                from hierarchical_atl import AbstractBrain
                
                self.brain = AbstractBrain(
                    feature_dim=64,
                    n_concepts=200,
                    visual_input_size=56,
                )
                
                checkpoint = torch.load(self.brain_path, map_location=DEVICE, weights_only=True)
                self.brain.visual.load_state_dict(checkpoint['visual_state'])
                
                print(f"Stream {self.config.stream_id}: Loaded brain")
            except Exception as e:
                print(f"Stream {self.config.stream_id}: Could not load brain: {e}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from source."""
        if self.config.source_type == 'synthetic':
            return self._generate_synthetic_frame()
        
        elif self.config.source_type == 'webcam':
            try:
                import cv2
                cap = cv2.VideoCapture(self.config.source_url)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                pass
        
        elif self.config.source_type == 'file':
            # For video files, would need to track position
            pass
        
        return None
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generate synthetic frame for testing."""
        canvas = np.ones((56, 56, 3), dtype=np.float32)
        
        # Add some random objects
        t = time.time()
        
        # Moving circle
        x = int(28 + 20 * np.sin(t * 0.5))
        y = int(28 + 20 * np.cos(t * 0.3))
        
        yy, xx = np.ogrid[:56, :56]
        mask = ((xx - x)**2 + (yy - y)**2) <= 25
        canvas[mask] = [1.0, 0.0, 0.0]  # Red
        
        # Random noise (simulates lighting changes)
        if np.random.random() < 0.1:
            canvas += np.random.randn(56, 56, 3) * 0.1
            canvas = np.clip(canvas, 0, 1)
        
        return (canvas * 255).astype(np.uint8)
    
    def process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert frame to activation."""
        frame_t = torch.from_numpy(frame).float() / 255.0
        frame_t = frame_t.permute(2, 0, 1).to(DEVICE)
        
        # Resize if needed
        if frame_t.shape[1] != 56 or frame_t.shape[2] != 56:
            frame_t = F.interpolate(
                frame_t.unsqueeze(0),
                size=(56, 56),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        
        if self.brain:
            activation = self.brain.get_scene_activation(frame_t)
        else:
            # Simple activation: just flatten and normalize
            activation = frame_t.flatten()
            activation = F.normalize(activation.unsqueeze(0), dim=1).squeeze(0)
        
        return activation
    
    def detect_event(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
    ) -> Optional[Tuple[str, float]]:
        """Detect event between frames."""
        change = current - previous
        magnitude = torch.norm(change).item()
        
        if magnitude < 0.05:
            return None  # No significant event
        
        # Classify event
        similarity = F.cosine_similarity(
            current.unsqueeze(0),
            previous.unsqueeze(0)
        ).item()
        
        if similarity > 0.8:
            event_type = "moved"
        elif similarity > 0.5:
            event_type = "changed"
        else:
            event_type = "major_change"
        
        return event_type, magnitude
    
    def run(self, duration_seconds: int = 3600):
        """Run observation loop."""
        print(f"Stream {self.config.stream_id}: Starting observation")
        
        self.load_brain()
        
        start_time = time.time()
        previous_activation = None
        
        while self.running and (time.time() - start_time) < duration_seconds:
            # Get frame
            frame = self.get_frame()
            
            if frame is None:
                time.sleep(self.config.sample_interval)
                continue
            
            # Process
            activation = self.process_frame(frame)
            self.frames_processed += 1
            
            # Detect event
            if previous_activation is not None:
                event_result = self.detect_event(activation, previous_activation)
                
                if event_result:
                    event_type, magnitude = event_result
                    
                    self.db.add_event(
                        stream_id=self.config.stream_id,
                        event_type=event_type,
                        change_magnitude=magnitude,
                        activation=activation,
                    )
                    
                    self.events_detected += 1
            
            previous_activation = activation
            
            # Update stats periodically
            if self.frames_processed % 100 == 0:
                self.db.update_stream_stats(
                    self.config.stream_id,
                    frames=100,
                    events=self.events_detected,
                )
                self.events_detected = 0
            
            # Sleep
            time.sleep(self.config.sample_interval)
        
        # Final update
        self.db.update_stream_stats(
            self.config.stream_id,
            frames=self.frames_processed % 100,
            events=self.events_detected,
            status="stopped",
        )
        
        print(f"Stream {self.config.stream_id}: Stopped. Processed {self.frames_processed} frames")
    
    def stop(self):
        """Stop observation."""
        self.running = False


class ContinuousObservationPipeline:
    """
    Coordinate multiple observation streams.
    
    Runs for days/weeks to accumulate events.
    """
    
    def __init__(self, db_path: str = 'observations.db'):
        self.db = ObservationDatabase(db_path)
        self.streams: Dict[str, StreamConfig] = {}
        self.observers: Dict[str, StreamObserver] = {}
        self.threads: Dict[str, threading.Thread] = {}
        self.running = False
    
    def add_stream(self, config: StreamConfig):
        """Add observation stream."""
        self.streams[config.stream_id] = config
    
    def start(self, duration_seconds: int = 3600):
        """Start all observation streams."""
        print(f"Starting {len(self.streams)} observation streams...")
        
        self.running = True
        
        for stream_id, config in self.streams.items():
            observer = StreamObserver(config, self.db.db_path)
            self.observers[stream_id] = observer
            
            thread = threading.Thread(
                target=observer.run,
                args=(duration_seconds,),
                daemon=True,
            )
            self.threads[stream_id] = thread
            thread.start()
        
        print("All streams started")
    
    def stop(self):
        """Stop all streams."""
        print("Stopping observation...")
        self.running = False
        
        for observer in self.observers.values():
            observer.stop()
        
        for thread in self.threads.values():
            thread.join(timeout=5)
        
        print("Observation stopped")
    
    def get_progress(self) -> Dict:
        """Get current progress."""
        return self.db.get_stats()
    
    def extract_routines(self) -> List[Dict]:
        """Extract learned routines."""
        return self.db.extract_routines()


def run_observation_demo(duration_minutes: int = 1):
    """Demo the continuous observation pipeline."""
    print("=" * 70)
    print("CONTINUOUS OBSERVATION DEMO")
    print(f"Running for {duration_minutes} minute(s)")
    print("=" * 70)
    
    # Create pipeline
    pipeline = ContinuousObservationPipeline('demo_observations.db')
    
    # Add synthetic streams
    streams = [
        StreamConfig(
            stream_id='traffic_cam_1',
            source_type='synthetic',
            source_url='',
            sample_interval=0.5,
            description='Simulated traffic camera',
        ),
        StreamConfig(
            stream_id='nature_cam_1',
            source_type='synthetic',
            source_url='',
            sample_interval=1.0,
            description='Simulated nature camera',
        ),
    ]
    
    for config in streams:
        pipeline.add_stream(config)
    
    # Start observation
    pipeline.start(duration_seconds=duration_minutes * 60)
    
    # Monitor progress
    try:
        for i in range(duration_minutes * 6):  # Check every 10 seconds
            time.sleep(10)
            stats = pipeline.get_progress()
            print(f"\n  Progress: {stats['total_events']} events, {stats['total_clusters']} clusters")
    except KeyboardInterrupt:
        pass
    
    # Stop
    pipeline.stop()
    
    # Final stats
    print("\n" + "=" * 60)
    print("OBSERVATION COMPLETE")
    print("=" * 60)
    
    stats = pipeline.get_progress()
    print(f"\n  Total events: {stats['total_events']}")
    print(f"  Total clusters: {stats['total_clusters']}")
    print(f"  Event types: {stats['event_types']}")
    
    # Extract routines
    routines = pipeline.extract_routines()
    print(f"\n  Routines found: {len(routines)}")
    for routine in routines[:5]:
        print(f"    - {routine['event_type']} at {routine['hour']}:00 ({routine['frequency']} times)")
    
    return pipeline


def check_observation_progress(db_path: str = 'observations.db'):
    """Check progress of running observation."""
    print("=" * 70)
    print("OBSERVATION PROGRESS CHECK")
    print("=" * 70)
    
    db = ObservationDatabase(db_path)
    stats = db.get_stats()
    
    print(f"\n  Total events: {stats['total_events']}")
    print(f"  Total clusters: {stats['total_clusters']}")
    print(f"  Anomalies: {stats['total_anomalies']}")
    
    print("\n  Streams:")
    for stream_id, stream_stats in stats['streams'].items():
        print(f"    {stream_id}: {stream_stats['frames']} frames, "
              f"{stream_stats['events']} events, status: {stream_stats['status']}")
    
    print("\n  Event types:")
    for event_type, count in stats['event_types'].items():
        print(f"    {event_type}: {count}")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', '-d', action='store_true',
                       help='Run demo observation')
    parser.add_argument('--check', '-c', action='store_true',
                       help='Check observation progress')
    parser.add_argument('--duration', type=int, default=1,
                       help='Duration in minutes for demo')
    parser.add_argument('--db', type=str, default='observations.db',
                       help='Database path')
    
    args = parser.parse_args()
    
    if args.check:
        check_observation_progress(args.db)
    else:
        run_observation_demo(args.duration)
