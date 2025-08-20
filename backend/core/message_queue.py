"""
Message Queue Infrastructure for RevOps Automation Platform.

This module provides Redis-based message queue capabilities for inter-service
communication, including retry logic, dead letter queues, monitoring, and
event publishing/subscription patterns.
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import structlog
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError

from backend.core.config import get_settings

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class MessageStatus(str, Enum):
    """Message processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"
    RETRYING = "retrying"


@dataclass
class QueueMessage:
    """Message structure for queue operations"""
    
    id: str
    queue_name: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = None
    scheduled_at: datetime = None
    expires_at: datetime = None
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 60.0  # seconds
    consumer_id: Optional[str] = None
    error_message: Optional[str] = None
    processing_duration: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.scheduled_at is None:
            self.scheduled_at = self.created_at
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for Redis storage"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key in ['created_at', 'scheduled_at', 'expires_at']:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueMessage':
        """Create message from dictionary"""
        # Convert ISO strings back to datetime objects
        for key in ['created_at', 'scheduled_at', 'expires_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def should_retry(self) -> bool:
        """Check if message should be retried"""
        return (
            self.status in [MessageStatus.FAILED, MessageStatus.RETRYING] and
            self.retry_count < self.max_retries and
            not self.is_expired()
        )
    
    def get_retry_delay(self) -> float:
        """Calculate retry delay with exponential backoff"""
        base_delay = self.retry_delay
        exponential_delay = base_delay * (2 ** self.retry_count)
        # Cap at 1 hour
        return min(exponential_delay, 3600.0)


class QueueConfig(BaseModel):
    """Queue configuration"""
    
    name: str
    redis_url: str = Field(default="redis://localhost:6379/0")
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=60.0, ge=1.0)
    dead_letter_ttl: int = Field(default=7 * 24 * 3600, ge=3600)  # 7 days
    message_ttl: int = Field(default=24 * 3600, ge=60)  # 24 hours
    batch_size: int = Field(default=10, ge=1, le=100)
    visibility_timeout: float = Field(default=300.0, ge=10.0)  # 5 minutes
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True)
    metrics_interval: float = Field(default=60.0, ge=10.0)
    
    # Consumer settings
    consumer_timeout: float = Field(default=30.0, ge=1.0)
    max_concurrent_messages: int = Field(default=5, ge=1)
    
    @validator('name')
    def validate_queue_name(cls, v):
        """Validate queue name format"""
        if not v or not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Queue name must contain only alphanumeric characters, hyphens, and underscores")
        return v


class MessageHandler(ABC):
    """Abstract base class for message handlers"""
    
    @abstractmethod
    async def handle_message(self, message: QueueMessage) -> bool:
        """
        Handle a message from the queue.
        
        Args:
            message: The message to process
            
        Returns:
            True if message was processed successfully, False otherwise
        """
        pass
    
    async def handle_error(self, message: QueueMessage, error: Exception):
        """
        Handle processing error. Override to implement custom error handling.
        
        Args:
            message: The message that failed to process
            error: The exception that occurred
        """
        logger.error(
            "message_processing_error",
            message_id=message.id,
            queue_name=message.queue_name,
            error=str(error),
            retry_count=message.retry_count
        )


class QueueMetrics:
    """Queue metrics collector"""
    
    def __init__(self, redis_client: Redis, queue_name: str):
        self.redis_client = redis_client
        self.queue_name = queue_name
        self.metrics_key = f"queue:metrics:{queue_name}"
    
    async def record_message_published(self):
        """Record a message published event"""
        await self._increment_counter("messages_published")
    
    async def record_message_consumed(self, processing_duration: float):
        """Record a message consumed event"""
        await self._increment_counter("messages_consumed")
        await self._record_duration("processing_duration", processing_duration)
    
    async def record_message_failed(self):
        """Record a message failed event"""
        await self._increment_counter("messages_failed")
    
    async def record_message_retried(self):
        """Record a message retry event"""
        await self._increment_counter("messages_retried")
    
    async def record_dead_letter(self):
        """Record a dead letter event"""
        await self._increment_counter("dead_letters")
    
    async def _increment_counter(self, metric_name: str):
        """Increment a counter metric"""
        try:
            key = f"{self.metrics_key}:{metric_name}"
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, 86400)  # Expire after 24 hours
        except RedisError as e:
            logger.warning("metrics_update_failed", metric=metric_name, error=str(e))
    
    async def _record_duration(self, metric_name: str, duration: float):
        """Record a duration metric (simple average for now)"""
        try:
            key = f"{self.metrics_key}:{metric_name}"
            # Use Redis hash to store sum and count for average calculation
            await self.redis_client.hincrbyfloat(key, "sum", duration)
            await self.redis_client.hincrby(key, "count", 1)
            await self.redis_client.expire(key, 86400)
        except RedisError as e:
            logger.warning("metrics_duration_failed", metric=metric_name, error=str(e))
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current queue metrics"""
        try:
            metrics = {}
            
            # Get counter metrics
            counter_keys = await self.redis_client.keys(f"{self.metrics_key}:*")
            for key in counter_keys:
                metric_name = key.split(":")[-1]
                if metric_name == "processing_duration":
                    # Handle duration metric
                    hash_data = await self.redis_client.hgetall(key)
                    if hash_data and hash_data.get("count", 0) > 0:
                        total_sum = float(hash_data.get("sum", 0))
                        count = int(hash_data.get("count", 0))
                        metrics[f"{metric_name}_avg"] = total_sum / count
                        metrics[f"{metric_name}_total"] = total_sum
                        metrics[f"{metric_name}_count"] = count
                else:
                    # Handle counter metric
                    value = await self.redis_client.get(key)
                    if value:
                        metrics[metric_name] = int(value)
            
            return metrics
            
        except RedisError as e:
            logger.error("metrics_retrieval_failed", error=str(e))
            return {}


class MessageQueue:
    """
    Redis-based message queue with retry logic, dead letter queues, and monitoring.
    
    Provides reliable message queuing capabilities for microservices communication
    with support for priorities, retries, dead letter handling, and metrics.
    """
    
    def __init__(self, config: QueueConfig):
        self.config = config
        self.redis_client: Optional[Redis] = None
        self.metrics: Optional[QueueMetrics] = None
        self._handlers: Dict[str, MessageHandler] = {}
        self._consumer_tasks: List[asyncio.Task] = []
        self._running = False
        
        # Queue key names
        self.main_queue_key = f"queue:{config.name}"
        self.processing_queue_key = f"queue:{config.name}:processing"
        self.dead_letter_queue_key = f"queue:{config.name}:dead_letter"
        self.scheduled_queue_key = f"queue:{config.name}:scheduled"
        
        logger.info("message_queue_initialized", queue_name=config.name)
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                decode_responses=True,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Initialize metrics
            if self.config.enable_metrics:
                self.metrics = QueueMetrics(self.redis_client, self.config.name)
            
            logger.info("redis_connected", queue_name=self.config.name)
            
        except Exception as e:
            logger.error("redis_connection_failed", queue_name=self.config.name, error=str(e))
            raise
    
    async def disconnect(self):
        """Disconnect from Redis and cleanup"""
        self._running = False
        
        # Cancel consumer tasks
        for task in self._consumer_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("redis_disconnected", queue_name=self.config.name)
    
    @asynccontextmanager
    async def connection(self):
        """Async context manager for Redis connection"""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()
    
    async def publish_message(
        self,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        delay: Optional[float] = None,
        expires_in: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Publish a message to the queue.
        
        Args:
            payload: Message payload
            priority: Message priority
            delay: Delay before message becomes available (seconds)
            expires_in: Message expiration time (seconds)
            **kwargs: Additional message properties
            
        Returns:
            Message ID
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        # Calculate scheduled time
        scheduled_at = now
        if delay and delay > 0:
            scheduled_at = now + timedelta(seconds=delay)
        
        # Calculate expiration time
        expires_at = None
        if expires_in and expires_in > 0:
            expires_at = now + timedelta(seconds=expires_in)
        elif not expires_in:
            expires_at = now + timedelta(seconds=self.config.message_ttl)
        
        # Create message
        message = QueueMessage(
            id=message_id,
            queue_name=self.config.name,
            payload=payload,
            priority=priority,
            created_at=now,
            scheduled_at=scheduled_at,
            expires_at=expires_at,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            **kwargs
        )
        
        # Determine target queue
        if delay and delay > 0:
            # Schedule for future delivery
            await self._schedule_message(message)
        else:
            # Add to main queue immediately
            await self._enqueue_message(message)
        
        # Record metrics
        if self.metrics:
            await self.metrics.record_message_published()
        
        logger.info(
            "message_published",
            message_id=message_id,
            queue_name=self.config.name,
            priority=priority.value,
            delayed=delay is not None,
            delay=delay
        )
        
        return message_id
    
    async def _enqueue_message(self, message: QueueMessage):
        """Add message to the main queue with priority support"""
        message_data = json.dumps(message.to_dict())
        
        # Use Redis sorted set for priority queuing
        # Higher priority = lower score (processed first)
        priority_scores = {
            MessagePriority.CRITICAL: 1,
            MessagePriority.HIGH: 2,
            MessagePriority.NORMAL: 3,
            MessagePriority.LOW: 4
        }
        
        score = priority_scores.get(message.priority, 3)
        
        await self.redis_client.zadd(
            self.main_queue_key,
            {message_data: score}
        )
    
    async def _schedule_message(self, message: QueueMessage):
        """Schedule a message for future delivery"""
        message_data = json.dumps(message.to_dict())
        timestamp = message.scheduled_at.timestamp()
        
        await self.redis_client.zadd(
            self.scheduled_queue_key,
            {message_data: timestamp}
        )
    
    async def consume_messages(self, handler: MessageHandler, consumer_id: str = None):
        """
        Start consuming messages from the queue.
        
        Args:
            handler: Message handler instance
            consumer_id: Optional consumer identifier
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        consumer_id = consumer_id or f"consumer-{uuid.uuid4()}"
        self._running = True
        
        logger.info("consumer_started", queue_name=self.config.name, consumer_id=consumer_id)
        
        # Start scheduled message processor
        scheduler_task = asyncio.create_task(self._process_scheduled_messages())
        self._consumer_tasks.append(scheduler_task)
        
        # Start main message processor
        processor_task = asyncio.create_task(
            self._process_messages(handler, consumer_id)
        )
        self._consumer_tasks.append(processor_task)
        
        # Start retry processor
        retry_task = asyncio.create_task(self._process_retries())
        self._consumer_tasks.append(retry_task)
        
        try:
            await asyncio.gather(*self._consumer_tasks)
        except asyncio.CancelledError:
            logger.info("consumer_cancelled", queue_name=self.config.name, consumer_id=consumer_id)
        except Exception as e:
            logger.error("consumer_error", queue_name=self.config.name, consumer_id=consumer_id, error=str(e))
            raise
    
    async def _process_messages(self, handler: MessageHandler, consumer_id: str):
        """Process messages from the main queue"""
        while self._running:
            try:
                # Get messages with priority (lowest score first)
                result = await self.redis_client.zpopmin(self.main_queue_key, 1)
                
                if not result:
                    # No messages available, wait a bit
                    await asyncio.sleep(1.0)
                    continue
                
                message_data, _ = result[0]
                message = QueueMessage.from_dict(json.loads(message_data))
                
                # Check if message expired
                if message.is_expired():
                    logger.warning("message_expired", message_id=message.id)
                    continue
                
                # Process message
                await self._handle_message(message, handler, consumer_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("message_processing_loop_error", error=str(e))
                await asyncio.sleep(5.0)  # Wait before retrying
    
    async def _handle_message(self, message: QueueMessage, handler: MessageHandler, consumer_id: str):
        """Handle a single message"""
        start_time = time.time()
        message.status = MessageStatus.PROCESSING
        message.consumer_id = consumer_id
        
        try:
            # Add to processing queue for visibility timeout
            await self._add_to_processing_queue(message)
            
            # Process the message
            success = await asyncio.wait_for(
                handler.handle_message(message),
                timeout=self.config.consumer_timeout
            )
            
            processing_duration = time.time() - start_time
            message.processing_duration = processing_duration
            
            if success:
                # Message processed successfully
                message.status = MessageStatus.COMPLETED
                await self._remove_from_processing_queue(message)
                
                if self.metrics:
                    await self.metrics.record_message_consumed(processing_duration)
                
                logger.info(
                    "message_processed",
                    message_id=message.id,
                    queue_name=self.config.name,
                    duration=processing_duration
                )
            else:
                # Message processing failed
                await self._handle_message_failure(message, "Handler returned False")
            
        except asyncio.TimeoutError:
            await self._handle_message_failure(message, "Processing timeout")
        except Exception as e:
            await self._handle_message_failure(message, str(e))
            await handler.handle_error(message, e)
    
    async def _handle_message_failure(self, message: QueueMessage, error_message: str):
        """Handle message processing failure"""
        message.error_message = error_message
        message.retry_count += 1
        
        await self._remove_from_processing_queue(message)
        
        if message.should_retry():
            # Schedule for retry
            message.status = MessageStatus.RETRYING
            retry_delay = message.get_retry_delay()
            message.scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=retry_delay)
            
            await self._schedule_message(message)
            
            if self.metrics:
                await self.metrics.record_message_retried()
            
            logger.warning(
                "message_retry_scheduled",
                message_id=message.id,
                retry_count=message.retry_count,
                retry_delay=retry_delay,
                error=error_message
            )
        else:
            # Move to dead letter queue
            message.status = MessageStatus.DEAD_LETTER
            await self._move_to_dead_letter(message)
            
            if self.metrics:
                await self.metrics.record_dead_letter()
            
            logger.error(
                "message_dead_letter",
                message_id=message.id,
                retry_count=message.retry_count,
                error=error_message
            )
        
        if self.metrics:
            await self.metrics.record_message_failed()
    
    async def _add_to_processing_queue(self, message: QueueMessage):
        """Add message to processing queue with visibility timeout"""
        timeout_timestamp = time.time() + self.config.visibility_timeout
        message_data = json.dumps(message.to_dict())
        
        await self.redis_client.zadd(
            self.processing_queue_key,
            {message_data: timeout_timestamp}
        )
    
    async def _remove_from_processing_queue(self, message: QueueMessage):
        """Remove message from processing queue"""
        message_data = json.dumps(message.to_dict())
        await self.redis_client.zrem(self.processing_queue_key, message_data)
    
    async def _move_to_dead_letter(self, message: QueueMessage):
        """Move message to dead letter queue"""
        message_data = json.dumps(message.to_dict())
        
        await self.redis_client.lpush(self.dead_letter_queue_key, message_data)
        await self.redis_client.expire(
            self.dead_letter_queue_key,
            self.config.dead_letter_ttl
        )
    
    async def _process_scheduled_messages(self):
        """Process scheduled messages and move them to main queue when ready"""
        while self._running:
            try:
                now = time.time()
                
                # Get messages scheduled for now or earlier
                result = await self.redis_client.zrangebyscore(
                    self.scheduled_queue_key,
                    0,
                    now,
                    start=0,
                    num=self.config.batch_size,
                    withscores=True
                )
                
                if not result:
                    await asyncio.sleep(10.0)  # Check every 10 seconds
                    continue
                
                for message_data, timestamp in result:
                    message = QueueMessage.from_dict(json.loads(message_data))
                    
                    # Remove from scheduled queue
                    await self.redis_client.zrem(self.scheduled_queue_key, message_data)
                    
                    # Check if expired
                    if message.is_expired():
                        logger.warning("scheduled_message_expired", message_id=message.id)
                        continue
                    
                    # Add to main queue
                    await self._enqueue_message(message)
                    
                    logger.debug("scheduled_message_promoted", message_id=message.id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("scheduled_processor_error", error=str(e))
                await asyncio.sleep(10.0)
    
    async def _process_retries(self):
        """Process messages that have timed out in processing queue"""
        while self._running:
            try:
                now = time.time()
                
                # Get messages that have timed out
                result = await self.redis_client.zrangebyscore(
                    self.processing_queue_key,
                    0,
                    now,
                    start=0,
                    num=self.config.batch_size,
                    withscores=True
                )
                
                for message_data, timeout_timestamp in result:
                    message = QueueMessage.from_dict(json.loads(message_data))
                    
                    # Remove from processing queue
                    await self.redis_client.zrem(self.processing_queue_key, message_data)
                    
                    # Handle as failure (timeout)
                    await self._handle_message_failure(message, "Visibility timeout exceeded")
                    
                    logger.warning("message_visibility_timeout", message_id=message.id)
                
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("retry_processor_error", error=str(e))
                await asyncio.sleep(30.0)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            stats = {
                "queue_name": self.config.name,
                "main_queue_size": await self.redis_client.zcard(self.main_queue_key),
                "processing_queue_size": await self.redis_client.zcard(self.processing_queue_key),
                "scheduled_queue_size": await self.redis_client.zcard(self.scheduled_queue_key),
                "dead_letter_queue_size": await self.redis_client.llen(self.dead_letter_queue_key),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if self.metrics:
                metrics = await self.metrics.get_metrics()
                stats["metrics"] = metrics
            
            return stats
            
        except Exception as e:
            logger.error("queue_stats_error", error=str(e))
            return {"error": str(e)}
    
    async def purge_queue(self):
        """Purge all messages from the queue (use with caution)"""
        await self.redis_client.delete(
            self.main_queue_key,
            self.processing_queue_key,
            self.scheduled_queue_key,
            self.dead_letter_queue_key
        )
        
        logger.warning("queue_purged", queue_name=self.config.name)


# Event system for pub/sub messaging

class EventMessage(BaseModel):
    """Event message for pub/sub"""
    
    event_type: str
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_service: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EventHandler(ABC):
    """Abstract event handler"""
    
    @abstractmethod
    async def handle_event(self, event: EventMessage) -> bool:
        """Handle an event"""
        pass


class EventBus:
    """Redis-based event bus for pub/sub messaging"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client: Optional[Redis] = None
        self.pubsub = None
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._running = False
    
    async def connect(self):
        """Connect to Redis"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        self.pubsub = self.redis_client.pubsub()
        
        logger.info("event_bus_connected")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        self._running = False
        
        if self.pubsub:
            await self.pubsub.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("event_bus_disconnected")
    
    async def publish_event(self, event: EventMessage):
        """Publish an event"""
        if not self.redis_client:
            raise RuntimeError("Event bus not connected")
        
        channel = f"events:{event.event_type}"
        message = event.json()
        
        await self.redis_client.publish(channel, message)
        
        logger.info(
            "event_published",
            event_type=event.event_type,
            event_id=event.event_id,
            source_service=event.source_service
        )
    
    def subscribe_to_event(self, event_type: str, handler: EventHandler):
        """Subscribe to an event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        
        self._handlers[event_type].append(handler)
        
        logger.info("event_subscription_added", event_type=event_type)
    
    async def start_listening(self):
        """Start listening for events"""
        if not self.pubsub:
            raise RuntimeError("Event bus not connected")
        
        # Subscribe to all registered event types
        for event_type in self._handlers.keys():
            channel = f"events:{event_type}"
            await self.pubsub.subscribe(channel)
        
        self._running = True
        
        logger.info("event_bus_listening", event_types=list(self._handlers.keys()))
        
        async for message in self.pubsub.listen():
            if not self._running:
                break
            
            if message['type'] != 'message':
                continue
            
            try:
                # Parse event
                event_data = json.loads(message['data'])
                event = EventMessage(**event_data)
                
                # Get handlers for this event type
                handlers = self._handlers.get(event.event_type, [])
                
                # Process handlers concurrently
                if handlers:
                    tasks = [handler.handle_event(event) for handler in handlers]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error("event_processing_error", error=str(e), message=message)


# Factory functions

def create_message_queue(
    queue_name: str,
    redis_url: str = None,
    **config_overrides
) -> MessageQueue:
    """
    Create a message queue instance.
    
    Args:
        queue_name: Name of the queue
        redis_url: Redis connection URL
        **config_overrides: Additional configuration overrides
        
    Returns:
        MessageQueue instance
    """
    settings = get_settings()
    
    config = QueueConfig(
        name=queue_name,
        redis_url=redis_url or "redis://localhost:6379/0",
        **config_overrides
    )
    
    return MessageQueue(config)


def create_event_bus(redis_url: str = None) -> EventBus:
    """
    Create an event bus instance.
    
    Args:
        redis_url: Redis connection URL
        
    Returns:
        EventBus instance
    """
    return EventBus(redis_url or "redis://localhost:6379/0")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    class TestMessageHandler(MessageHandler):
        async def handle_message(self, message: QueueMessage) -> bool:
            print(f"Processing message {message.id}: {message.payload}")
            await asyncio.sleep(1)  # Simulate processing
            return True
    
    class TestEventHandler(EventHandler):
        async def handle_event(self, event: EventMessage) -> bool:
            print(f"Received event {event.event_type}: {event.payload}")
            return True
    
    async def test_message_queue():
        """Test message queue functionality"""
        queue = create_message_queue("test-queue")
        
        async with queue.connection():
            # Publish some test messages
            await queue.publish_message(
                {"action": "test", "data": "hello world"},
                priority=MessagePriority.HIGH
            )
            
            # Get stats
            stats = await queue.get_queue_stats()
            print(f"Queue stats: {stats}")
            
            # Start consumer (would run indefinitely in real usage)
            handler = TestMessageHandler()
            # await queue.consume_messages(handler)
    
    async def test_event_bus():
        """Test event bus functionality"""
        event_bus = create_event_bus()
        
        await event_bus.connect()
        
        # Subscribe to events
        handler = TestEventHandler()
        event_bus.subscribe_to_event("test.event", handler)
        
        # Publish an event
        event = EventMessage(
            event_type="test.event",
            source_service="test-service",
            payload={"message": "hello events"}
        )
        
        await event_bus.publish_event(event)
        
        await event_bus.disconnect()
    
    # Run tests
    print("Testing message queue...")
    asyncio.run(test_message_queue())
    
    print("Testing event bus...")
    asyncio.run(test_event_bus())