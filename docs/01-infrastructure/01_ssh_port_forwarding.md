# SSH Port Forwarding & Dashboard Access Guide

## Current Status

### Spec-Workflow Dashboard
- **Status**: Currently running in headless mode (no web interface)
- **Approvals**: Handled directly in conversation, not via dashboard
- **Future**: May add web dashboard that would require port forwarding

## Port Requirements for RevOps Platform

### Services and Default Ports

| Service | Default Port | Purpose | SSH Forward Command |
|---------|-------------|---------|-------------------|
| PostgreSQL | 5432 | Local database | `ssh -L 5432:localhost:5432 user@server` |
| BGE Service | 8080 | BGE-M3 embeddings API | `ssh -L 8080:localhost:8080 user@server` |
| Backend API | 8000 | FastAPI backend | `ssh -L 8000:localhost:8000 user@server` |
| Frontend Dev | 3000 | React development server | `ssh -L 3000:localhost:3000 user@server` |
| Frontend Build | 5173 | Vite production preview | `ssh -L 5173:localhost:5173 user@server` |

## SSH Port Forwarding Setup

### Single Port Forwarding
```bash
# Forward a single port
ssh -L [local_port]:localhost:[remote_port] [username]@[server]

# Example: Forward FastAPI backend
ssh -L 8000:localhost:8000 loyd2888@revops-server
```

### Multiple Port Forwarding
```bash
# Forward multiple ports in one command
ssh -L 5432:localhost:5432 \
    -L 8080:localhost:8080 \
    -L 8000:localhost:8000 \
    -L 3000:localhost:3000 \
    loyd2888@revops-server
```

### Persistent Connection with Config File
Add to `~/.ssh/config`:
```
Host revops-server
    HostName [server-ip-or-hostname]
    User loyd2888
    LocalForward 5432 localhost:5432
    LocalForward 8080 localhost:8080
    LocalForward 8000 localhost:8000
    LocalForward 3000 localhost:3000
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then connect with:
```bash
ssh revops-server
```

## Accessing Services from Local Desktop

Once SSH port forwarding is established:

### Database Access
```bash
# Connect to PostgreSQL from local machine
psql -h localhost -p 5432 -U postgres -d revops_core
```

### API Access
```bash
# Access FastAPI backend
curl http://localhost:8000/api/v1/health

# View API documentation
open http://localhost:8000/docs
```

### Frontend Access
```bash
# Access React development server
open http://localhost:3000

# Access Vite preview server
open http://localhost:5173
```

### BGE Service Access
```bash
# Test BGE embeddings service
curl -X POST http://localhost:8080/embeddings \
  -H "Content-Type: application/json" \
  -d '{"text": "test embedding"}'
```

## Docker Compose Considerations

If using Docker, ensure services bind to `0.0.0.0` not just `127.0.0.1`:

```yaml
# docker-compose.yml
services:
  postgres:
    ports:
      - "0.0.0.0:5432:5432"  # Accessible via SSH tunnel
  
  bge:
    ports:
      - "0.0.0.0:8080:8080"  # Accessible via SSH tunnel
  
  backend:
    ports:
      - "0.0.0.0:8000:8000"  # Accessible via SSH tunnel
```

## Troubleshooting

### Check if Port is Already in Use
```bash
# On local machine
lsof -i :8000

# On remote server
sudo netstat -tlnp | grep :8000
```

### Kill Existing SSH Tunnels
```bash
# Find SSH processes with port forwarding
ps aux | grep ssh | grep " -L "

# Kill specific process
kill -9 [PID]
```

### Test Connection
```bash
# Test if tunnel is working
nc -zv localhost 8000
```

## Security Considerations

1. **Use SSH keys** instead of passwords for authentication
2. **Limit forwarding** to specific IPs when possible
3. **Use firewall rules** on the server to restrict direct access
4. **Monitor connections** with `netstat` or `ss` commands
5. **Use VPN** as an alternative for production environments

## Alternative: Using ngrok (for Development)

For temporary public access during development:

```bash
# Install ngrok
brew install ngrok  # macOS
# or download from https://ngrok.com

# Expose local port
ngrok http 8000

# This provides a public URL like:
# https://abc123.ngrok.io → localhost:8000
```

## Future Dashboard Considerations

When the spec-workflow dashboard becomes available:
1. It will likely run on port 3001 or 8001
2. Add to SSH config: `LocalForward 3001 localhost:3001`
3. Access via: `http://localhost:3001/dashboard`

## Notes

- The current spec-workflow system runs in headless mode (no dashboard required)
- All approvals happen in the conversation, not via web interface
- Port forwarding will become essential when we deploy the full application stack
- Consider using tmux or screen to maintain persistent SSH sessions