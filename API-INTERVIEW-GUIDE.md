# REST APIs · FastAPI · gRPC — Ultimate Interview Preparation Guide

> **Audience:** AI Engineers, ML Engineers, Backend Engineers, Platform Engineers, MLOps Engineers, Software Engineers
> **Goal:** After studying this document you can confidently answer Software Engineering, AI Engineering, Backend, Distributed Systems, Production Architecture, and Design interviews.
> **Approach:** Every concept is covered with Theory → Diagram → Python → Go → Production Example → Pitfalls → Interview Questions.

---

## Table of Contents

1. [Introduction — APIs, Client-Server, Distributed Systems](#1-introduction)
2. [REST API Complete Guide](#2-rest-api-complete-guide)
3. [FastAPI Complete Guide](#3-fastapi-complete-guide)
4. [gRPC Complete Guide](#4-grpc-complete-guide)
5. [REST vs FastAPI vs gRPC — Decision Matrix](#5-rest-vs-fastapi-vs-grpc)
6. [System Design Perspective](#6-system-design-perspective)
7. [AI Engineering Perspective](#7-ai-engineering-perspective)
8. [Real Production Case Studies](#8-real-production-case-studies)
9. [Failure Scenarios and Remediation](#9-failure-scenarios)
10. [Interview Crash Revision — 300 Questions with Answers](#10-interview-crash-revision)
11. [Learning Roadmap — Beginner to Principal Engineer](#11-learning-roadmap)

---

# 1. Introduction

## 1.1 What is an API?

An **Application Programming Interface (API)** is a contract that defines how software components communicate. It specifies:
- **What** operations are available
- **What** data formats are used
- **What** errors can occur
- **What** authentication is required

APIs abstract away implementation details. The caller does not need to know how a service works — only how to call it.

### Why APIs Exist

| Problem without APIs | Solution with APIs |
|----------------------|-------------------|
| Every integration requires custom code | Standardized contract everyone can consume |
| Tight coupling between systems | Loose coupling — each side can evolve independently |
| Hard to scale individual components | Each service behind an API scales independently |
| Cannot expose functionality externally | APIs enable business models (Stripe, Twilio, OpenAI) |

## 1.2 Client-Server Architecture

```mermaid
flowchart LR
    Client["Client\n(Browser / Mobile / CLI / Service)"] -- "Request (HTTP/gRPC/WS)" --> Server["Server\n(API Service)"]
    Server -- "Response (JSON/Protobuf/HTML)" --> Client
    Server -- "Reads/Writes" --> DB[("Database")]
    Server -- "Calls" --> Downstream["Downstream Services"]
```

**Rules of client-server:**
- Client initiates all requests
- Server processes and responds
- Neither knows the internal implementation of the other
- They communicate only through the agreed interface

## 1.3 Request-Response Model

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: Request (method + URL + headers + body)
    Note over S: Authenticate, validate, process
    S->>C: Response (status code + headers + body)
```

Every HTTP interaction is:
1. Client sends: **method** (GET/POST…) + **URL** + **headers** + optional **body**
2. Server responds: **status code** + **headers** + optional **body**
3. Connection may be kept alive for subsequent requests (HTTP/1.1 Keep-Alive, HTTP/2 multiplexing)

## 1.4 Distributed Systems Overview

A **distributed system** is a collection of independent computers that appear to users as a single coherent system.

**Key properties (CAP Theorem):**

| Property | Definition | Trade-off |
|----------|-----------|-----------|
| **Consistency** | All nodes return the same data at the same time | Must block writes during partition |
| **Availability** | Every request receives a response (not necessarily latest data) | May return stale data |
| **Partition Tolerance** | System continues operating despite network partitions | Required in any real distributed system |

> CAP: You can only guarantee 2 of 3 during a network partition. Real systems choose CP (e.g., HBase, ZooKeeper) or AP (e.g., Cassandra, DynamoDB).

**Eight fallacies of distributed systems** (every interviewer loves these):
1. The network is reliable
2. Latency is zero
3. Bandwidth is infinite
4. The network is secure
5. Topology doesn't change
6. There is one administrator
7. Transport cost is zero
8. The network is homogeneous

## 1.5 Monolith vs Microservices

```mermaid
flowchart TD
    subgraph mono [Monolith]
        UI_M["UI Layer"] --> BL["Business Logic"]
        BL --> DAL["Data Access Layer"]
        DAL --> DB_M[("Single DB")]
    end
    subgraph micro [Microservices]
        GW["API Gateway"] --> UserSvc["User Service"]
        GW --> OrderSvc["Order Service"]
        GW --> PaySvc["Payment Service"]
        UserSvc --> DB1[("Users DB")]
        OrderSvc --> DB2[("Orders DB")]
        PaySvc --> DB3[("Payments DB")]
    end
```

| Dimension | Monolith | Microservices |
|-----------|---------|---------------|
| Deployment | Single artifact | Many independent services |
| Scaling | Scale entire app | Scale individual services |
| Development | Simple at small scale | Complex orchestration |
| Fault isolation | One bug can crash all | Failures are isolated |
| Technology | Single stack | Polyglot |
| Data | Shared database | Database per service |
| Communication | In-process function calls | Network (REST/gRPC/messages) |
| Latency | None (in-process) | Network overhead |
| Best for | Early-stage products, small teams | Large teams, independent scaling needs |

**Interview answer on when to choose microservices:**
> "I would choose microservices when different components have very different scaling requirements (e.g., inference is 100× heavier than auth), when multiple large teams own separate domains and need independent deploy cadences, or when different parts of the system need different SLAs. I would NOT start with microservices — I would start with a well-structured monolith and extract services when pain points emerge."

---

# 2. REST API Complete Guide

## 2.1 History and the Richardson Maturity Model

REST (Representational State Transfer) was defined by Roy Fielding in his 2000 PhD dissertation. It is an **architectural style**, not a protocol.

### Richardson Maturity Model

```mermaid
flowchart TD
    L0["Level 0: The Swamp of POX\nSingle endpoint, action in body\nPOST /api {action: getUser}"] --> L1
    L1["Level 1: Resources\nSeparate endpoints per resource\nPOST /users, POST /orders"] --> L2
    L2["Level 2: HTTP Verbs\nGET /users, POST /users, DELETE /users/1"] --> L3
    L3["Level 3: Hypermedia (HATEOAS)\nResponses include links to related actions"]
```

| Level | Name | Example | Maturity |
|-------|------|---------|---------|
| 0 | POX (Plain Old XML/JSON) | `POST /api` with action in body | Poor |
| 1 | Resources | `POST /users`, `POST /orders` | Basic |
| 2 | HTTP Verbs | `GET /users`, `DELETE /users/1` | Good — most real APIs |
| 3 | HATEOAS | Response includes `_links` | Ideal — rarely implemented fully |

**Most production REST APIs are Level 2.** Level 3 (HATEOAS) adds discoverability but is complex and rarely used outside specific domains.

## 2.2 REST Constraints

### Constraint 1: Client-Server Separation
**Definition:** UI and data storage are separated. Client and server evolve independently.
**Interview angle:** "This enables independent scaling and allows the same server to serve web, mobile, CLI clients."

### Constraint 2: Statelessness
**Definition:** Each request must contain all information needed to process it. Server holds NO session state between requests.

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: GET /orders Authorization: Bearer <token>
    Note over S: Token contains all identity info — no session lookup needed
    S->>C: 200 OK [{orders}]
    C->>S: POST /orders Authorization: Bearer <token>
    Note over S: Again, token is self-contained — no memory of previous request
    S->>C: 201 Created
```

**Why statelessness matters:** Horizontal scaling is trivial — any server can handle any request. No sticky sessions needed.
**Cost:** Larger request payloads (must include auth on every request).

### Constraint 3: Cacheability
**Definition:** Responses must label themselves as cacheable or non-cacheable via HTTP headers (`Cache-Control`, `ETag`, `Expires`).
**Interview angle:** "Caching reduces server load and improves latency. Incorrect caching leads to stale data — you must explicitly set `Cache-Control: no-store` for sensitive data."

### Constraint 4: Uniform Interface
Four sub-constraints:
1. **Resource identification** — resources are identified by URIs (`/users/42`)
2. **Manipulation through representations** — client receives a representation (JSON) and can manipulate the resource by sending it back modified
3. **Self-descriptive messages** — `Content-Type: application/json` tells the client how to parse the body
4. **HATEOAS** — hypermedia links in responses drive state transitions

### Constraint 5: Layered System
**Definition:** Client cannot tell if it's talking to the actual server or an intermediary (load balancer, cache, gateway). Each layer only knows about its immediate neighbor.

### Constraint 6: Code on Demand (optional)
**Definition:** Server can send executable code to client (JavaScript). The only optional constraint.

## 2.3 REST Fundamentals

### Resources, URIs, URLs, URNs

| Term | Definition | Example |
|------|-----------|---------|
| **Resource** | The noun being accessed — a thing, not an action | "a user", "an order", "a model" |
| **URI** (Identifier) | Identifies a resource (may not be locatable) | `urn:isbn:0451450523` |
| **URL** (Locator) | URI that also specifies how to access it | `https://api.example.com/users/42` |
| **URN** (Name) | URI that names a resource permanently | `urn:uuid:6e8bc430-9c3a-11d9-9669` |
| **Endpoint** | A specific URL + HTTP method combination | `GET /users/{id}` |

**All URLs are URIs; not all URIs are URLs.**

### HTTP vs HTTPS

| Dimension | HTTP | HTTPS |
|-----------|------|-------|
| Port | 80 | 443 |
| Encryption | None — plaintext | TLS — encrypted |
| Authentication | None | Certificate validates server identity |
| Performance | Slightly faster | TLS handshake overhead (one-time with session resumption) |
| Production | Never for production | Always |

## 2.4 HTTP Methods Deep Dive

| Method | Purpose | Idempotent | Safe | Has Body |
|--------|---------|-----------|------|---------|
| GET | Retrieve resource | Yes | Yes | No |
| POST | Create resource / trigger action | No | No | Yes |
| PUT | Replace resource entirely | Yes | No | Yes |
| PATCH | Partially update resource | No* | No | Yes |
| DELETE | Delete resource | Yes | No | No |
| HEAD | GET headers only (no body) | Yes | Yes | No |
| OPTIONS | List allowed methods (CORS preflight) | Yes | Yes | No |
| TRACE | Echo request (diagnostic) | Yes | Yes | No |
| CONNECT | Tunnel (HTTPS proxy) | — | No | — |

*PATCH idempotency depends on implementation — JSON Patch is idempotent; custom patch semantics may not be.

**Idempotent:** Multiple identical requests produce the same result as a single request.
**Safe:** Request does not modify server state.

### Python Examples

```python
import httpx  # production-grade async HTTP client

# GET — retrieve a resource
async with httpx.AsyncClient(base_url="https://api.example.com") as client:
    response = await client.get("/users/42", headers={"Authorization": "Bearer <token>"})
    user = response.json()

# POST — create a resource
response = await client.post(
    "/users",
    json={"name": "Alice", "email": "alice@example.com"},
    headers={"Authorization": "Bearer <token>"},
)
assert response.status_code == 201
new_user = response.json()

# PUT — replace entire resource
response = await client.put(
    "/users/42",
    json={"name": "Alice Updated", "email": "alice@example.com", "role": "admin"},
)

# PATCH — partial update
response = await client.patch(
    "/users/42",
    json={"role": "admin"},   # only the fields to change
)

# DELETE — remove resource
response = await client.delete("/users/42")
assert response.status_code == 204  # No Content
```

### Go Examples

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

func main() {
    client := &http.Client{}

    // GET
    req, _ := http.NewRequest("GET", "https://api.example.com/users/42", nil)
    req.Header.Set("Authorization", "Bearer <token>")
    resp, _ := client.Do(req)
    defer resp.Body.Close()

    // POST
    body, _ := json.Marshal(map[string]string{"name": "Alice", "email": "alice@example.com"})
    req, _ = http.NewRequest("POST", "https://api.example.com/users", bytes.NewBuffer(body))
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer <token>")
    resp, _ = client.Do(req)
    fmt.Println(resp.StatusCode) // 201

    // PATCH
    patch, _ := json.Marshal(map[string]string{"role": "admin"})
    req, _ = http.NewRequest("PATCH", "https://api.example.com/users/42", bytes.NewBuffer(patch))
    req.Header.Set("Content-Type", "application/json")
    resp, _ = client.Do(req)

    // DELETE
    req, _ = http.NewRequest("DELETE", "https://api.example.com/users/42", nil)
    resp, _ = client.Do(req)
    fmt.Println(resp.StatusCode) // 204
}
```

## 2.5 HTTP Status Codes

### 1xx — Informational

| Code | Name | Meaning |
|------|------|---------|
| 100 | Continue | Client can send the request body |
| 101 | Switching Protocols | Server upgrading to WebSocket |

### 2xx — Success

| Code | Name | When to Use |
|------|------|------------|
| **200** | OK | Successful GET, PUT, PATCH — with body |
| **201** | Created | Successful POST — include `Location` header |
| **202** | Accepted | Request accepted for async processing — return job ID |
| **204** | No Content | Successful DELETE or action with no response body |

### 3xx — Redirection

| Code | Name | When to Use |
|------|------|------------|
| **301** | Moved Permanently | Resource URL changed forever — update bookmarks |
| **302** | Found | Temporary redirect |
| **304** | Not Modified | Client cache is still valid (ETag matched) |

### 4xx — Client Errors

| Code | Name | When to Use |
|------|------|------------|
| **400** | Bad Request | Malformed syntax, missing required field |
| **401** | Unauthorized | Missing or invalid authentication credentials |
| **403** | Forbidden | Authenticated but not authorized for this resource |
| **404** | Not Found | Resource does not exist |
| **405** | Method Not Allowed | HTTP method not supported for this endpoint |
| **409** | Conflict | State conflict — e.g., duplicate email, optimistic lock failure |
| **415** | Unsupported Media Type | `Content-Type` not supported by server |
| **422** | Unprocessable Entity | Syntax correct but semantically invalid (FastAPI validation errors) |
| **429** | Too Many Requests | Rate limit exceeded — include `Retry-After` header |

### 5xx — Server Errors

| Code | Name | When to Use |
|------|------|------------|
| **500** | Internal Server Error | Unhandled exception — never expose stack traces |
| **502** | Bad Gateway | Proxy received invalid response from upstream |
| **503** | Service Unavailable | Service overloaded or in maintenance |
| **504** | Gateway Timeout | Upstream took too long to respond |

**Interview questions on status codes:**
- "What's the difference between 401 and 403?" → 401 = not authenticated (don't know who you are); 403 = authenticated but not permitted (know who you are, but you can't do this)
- "When do you use 202?" → Async jobs — model training, video encoding, report generation
- "When do you return 409 vs 400?" → 409 for state conflicts (duplicate, stale version); 400 for malformed input

## 2.6 HTTP Headers

### Request Headers

| Header | Purpose | Production Example |
|--------|---------|-------------------|
| `Authorization` | Carry credentials | `Authorization: Bearer eyJhbG...` |
| `Content-Type` | Body format of request | `Content-Type: application/json` |
| `Accept` | Preferred response format | `Accept: application/json` |
| `User-Agent` | Client identification | `User-Agent: MyApp/2.1 Python/3.12` |
| `Idempotency-Key` | Deduplicate retried requests | `Idempotency-Key: uuid-v4` |
| `X-Correlation-ID` | Trace request across services | `X-Correlation-ID: req-abc123` |
| `X-Request-ID` | Unique ID for this request | `X-Request-ID: 550e8400-e29b-41d4-a716` |

### Response Headers

| Header | Purpose | Production Example |
|--------|---------|-------------------|
| `Cache-Control` | Caching directive | `Cache-Control: max-age=3600, private` |
| `ETag` | Resource version fingerprint | `ETag: "33a64df551"` |
| `If-None-Match` | Conditional GET — only respond if changed | `If-None-Match: "33a64df551"` |
| `If-Match` | Optimistic locking — only update if version matches | `If-Match: "33a64df551"` |
| `Location` | URL of newly created resource | `Location: /users/42` |
| `Retry-After` | How long before retrying (rate limit/503) | `Retry-After: 60` |
| `X-RateLimit-Remaining` | Remaining requests in window | `X-RateLimit-Remaining: 47` |
| `Strict-Transport-Security` | Force HTTPS | `HSTS: max-age=31536000; includeSubDomains` |

### ETag Caching Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: GET /users/42
    S->>C: 200 OK  ETag: "abc123"  {user data}
    Note over C: Store ETag
    C->>S: GET /users/42  If-None-Match: "abc123"
    Note over S: Resource unchanged
    S->>C: 304 Not Modified (no body)
    Note over C: Use cached data, save bandwidth
    Note over S: User updated elsewhere
    C->>S: GET /users/42  If-None-Match: "abc123"
    S->>C: 200 OK  ETag: "def456"  {updated user data}
```

## 2.7 Authentication and Authorization

**Authentication:** Who are you?
**Authorization:** What are you allowed to do?

### API Key

```
GET /v1/completions
Authorization: Bearer sk-...
```

**Flow:** Client includes static key in every request. Server looks up key in DB or cache.
**Pros:** Simple, stateless. **Cons:** No expiry, hard to rotate, must be kept secret.
**Use when:** Server-to-server M2M, simple developer APIs.

### JWT (JSON Web Token)

```mermaid
flowchart LR
    Client -- "POST /auth/login {user, pass}" --> AuthSvc["Auth Service"]
    AuthSvc -- "Sign JWT with secret" --> Token["JWT\nheader.payload.signature"]
    Token -- "Return to client" --> Client
    Client -- "GET /api/data\nAuthorization: Bearer JWT" --> APIServer["API Server"]
    APIServer -- "Verify signature locally\n(no DB lookup)" --> APIServer
    APIServer -- "200 OK data" --> Client
```

**JWT Structure:**
```
base64(header).base64(payload).HMAC_SHA256(header+payload, secret)
```

Header: `{"alg": "HS256", "typ": "JWT"}`
Payload: `{"sub": "user_42", "exp": 1735689600, "roles": ["admin"]}`

**Pros:** Stateless — no session store needed; self-contained; scales horizontally.
**Cons:** Cannot revoke before expiry; payload visible (encode, not encrypt); large token size.
**Mitigation:** Short expiry (15 min access token) + long refresh token; maintain a revocation list for critical paths.

### OAuth2

```mermaid
sequenceDiagram
    participant U as User
    participant C as Client App
    participant AS as Auth Server
    participant RS as Resource Server
    U->>C: Click "Login with Google"
    C->>AS: Redirect to /authorize?client_id=...&scope=email&response_type=code
    AS->>U: Show consent screen
    U->>AS: Approve
    AS->>C: Redirect back with authorization_code
    C->>AS: POST /token {code, client_secret}
    AS->>C: access_token + refresh_token
    C->>RS: GET /userinfo Authorization: Bearer access_token
    RS->>C: {email, name}
```

**Four grant types:**

| Grant | Use Case |
|-------|---------|
| Authorization Code | Web/mobile apps with user login (most common) |
| Client Credentials | M2M — no user involved |
| Device Code | TV/CLI devices with limited input |
| Implicit | Deprecated — do not use |

### mTLS (Mutual TLS)

In standard TLS, only the server presents a certificate. In mTLS, **both client and server** present certificates.

**Use when:** Service-to-service internal communication; zero-trust networks; financial APIs.

```mermaid
sequenceDiagram
    participant C as Client Service
    participant S as Server Service
    C->>S: Client Hello (supported ciphers)
    S->>C: Server Certificate
    S->>C: Certificate Request (asking for client cert)
    C->>S: Client Certificate
    C->>S: Signed data with client private key
    Note over C,S: Both sides verified — encrypted channel established
```

### RBAC vs ABAC

| Model | How it works | Example |
|-------|-------------|---------|
| **RBAC** (Role-Based) | Assign roles to users; roles have permissions | `admin` can DELETE; `viewer` can only GET |
| **ABAC** (Attribute-Based) | Evaluate policies based on attributes (user, resource, environment) | `allow if user.department == resource.department AND time.hour < 18` |

RBAC is simpler to implement. ABAC is more expressive for complex access patterns.

## 2.8 API Versioning

| Strategy | Example | Pros | Cons |
|----------|---------|------|------|
| **URI versioning** | `/v1/users` | Most visible, easy to route, cached separately | Pollutes URI design |
| **Header versioning** | `API-Version: 2` | Clean URIs | Harder to test in browser |
| **Query param** | `/users?version=2` | Easy to add | Easy to ignore in routing |
| **Media type (content negotiation)** | `Accept: application/vnd.api+json;version=2` | Most RESTful | Complex, rarely supported by tooling |

**Production recommendation:** URI versioning (`/v1/`, `/v2/`) for external APIs — it's explicit, cacheable, and easily routable at the gateway. Header versioning for internal APIs.

**Deprecation policy:**
1. Announce deprecation date 6+ months in advance
2. Return `Deprecation` header on old versions
3. Log all traffic to old version to identify remaining clients
4. Sunset (remove) only after traffic drops to zero

## 2.9 REST Design Principles

### Naming Conventions

| Rule | Wrong | Right |
|------|-------|-------|
| Use nouns, not verbs | `GET /getUser` | `GET /users/{id}` |
| Use plural nouns | `/user/42` | `/users/42` |
| Lowercase only | `/UserProfile` | `/user-profile` |
| Hyphens, not underscores | `/user_profile` | `/user-profile` |
| No file extensions | `/users.json` | `/users` (use Accept header) |

### Nested Resources

```
GET /users/{userId}/orders              # Orders belonging to a user
GET /users/{userId}/orders/{orderId}    # Specific order of a user
POST /users/{userId}/orders             # Create order for user
```

Rule: Nest only one level deep. Deeper nesting becomes unwieldy.

### Pagination

```python
# Offset-based (simple, standard)
GET /users?page=3&per_page=20
# Response: { "data": [...], "total": 1000, "page": 3, "per_page": 20 }

# Cursor-based (for real-time data, no page drift)
GET /users?cursor=eyJpZCI6MTAwfQ&limit=20
# Response: { "data": [...], "next_cursor": "eyJpZCI6MTIwfQ" }
```

**Cursor-based pagination** is preferred for production: consistent results even if data changes during pagination; constant-time DB queries regardless of page number.

### Filtering, Sorting, Searching

```
GET /orders?status=pending&created_after=2025-01-01     # filtering
GET /orders?sort=-created_at,amount                      # sort by created_at DESC, amount ASC
GET /users?search=alice                                   # full-text search
GET /users?fields=id,name,email                          # sparse fieldsets (projection)
```

### Idempotency Keys

For non-idempotent operations (POST), clients should send an `Idempotency-Key`:

```python
# Client sends unique key with POST
response = client.post(
    "/payments",
    json={"amount": 100, "to": "user_42"},
    headers={"Idempotency-Key": "a1b2c3d4-e5f6-..."},
)
# Server stores key + result. If retry with same key → return same result without re-processing
```

### Rate Limiting

```mermaid
flowchart LR
    Request --> RateLimiter["Rate Limiter\n(Token Bucket / Sliding Window)"]
    RateLimiter -- "Under limit" --> Handler["Request Handler"]
    RateLimiter -- "Exceeded" --> Response_429["429 Too Many Requests\nRetry-After: 60"]
```

**Algorithms:**
| Algorithm | How it works | Burst handling |
|-----------|-------------|----------------|
| Token Bucket | Tokens refill at fixed rate; each request consumes a token | Allows bursts up to bucket size |
| Leaky Bucket | Requests processed at fixed rate; excess queued or dropped | Smooths bursts |
| Fixed Window | Count requests per fixed time window | Window boundary bursts |
| Sliding Window | Rolling window; no boundary spike | Most accurate, higher memory |

## 2.10 REST Production Challenges

| Challenge | Symptoms | Mitigation |
|-----------|----------|-----------|
| **Thundering Herd** | Cache expires, all clients hit DB simultaneously | Jitter on cache expiry; cache warming; probabilistic early expiration |
| **Retry storms** | Client retries aggressively after 503, making overload worse | Exponential backoff with jitter; `Retry-After` header |
| **Duplicate requests** | Payment processed twice due to timeout retry | Idempotency keys with server-side deduplication |
| **Race conditions** | Two clients update same record simultaneously | Optimistic locking with ETags; database row-level locks |
| **N+1 queries** | Fetching 100 users + 100 separate DB calls for each user's orders | Eager loading; DataLoader pattern; denormalization |
| **Partial failures** | Payment charged but order not created | Saga pattern; compensating transactions; outbox pattern |
| **Large payloads** | 50MB JSON response slowing network | Pagination; streaming; compression (`Accept-Encoding: gzip`) |
| **Connection exhaustion** | Too many open connections to DB | Connection pooling; backpressure |

---

# 3. FastAPI Complete Guide

## 3.1 What is FastAPI?

FastAPI is a modern Python web framework for building APIs with:
- **Automatic OpenAPI/Swagger docs** from code
- **Pydantic v2** for request/response validation and serialization
- **Async-first** using Python's `asyncio` / ASGI
- **Type hints** as the primary interface — declare types, get validation + docs + IDE support

```mermaid
flowchart TD
    TypeHints["Python Type Hints"] --> Pydantic["Pydantic v2\nValidation & Serialization"]
    TypeHints --> OpenAPI["OpenAPI Schema\nAuto-generated docs"]
    ASGI["ASGI Server\n(Uvicorn / Hypercorn)"] --> Starlette["Starlette\nRouting, Middleware, WebSocket"]
    Starlette --> FastAPI["FastAPI"]
    Pydantic --> FastAPI
    OpenAPI --> FastAPI
```

### FastAPI vs Flask vs Django

| Dimension | FastAPI | Flask | Django |
|-----------|---------|-------|--------|
| **Performance** | High (ASGI, async) | Medium (WSGI sync) | Medium (WSGI, async since 3.1) |
| **Validation** | Automatic (Pydantic) | Manual | Serializers (DRF) |
| **Docs** | Auto OpenAPI/Swagger | Manual | DRF browsable API |
| **Async** | First-class | Partial (quart) | Partial |
| **ORM** | None (bring your own) | None | Django ORM built-in |
| **Learning curve** | Low | Very low | High |
| **Best for** | APIs, ML serving, async workloads | Simple APIs, scripts | Full-stack web apps |

## 3.2 ASGI vs WSGI

| | WSGI | ASGI |
|---|------|------|
| **Stands for** | Web Server Gateway Interface | Asynchronous Server Gateway Interface |
| **Introduced** | PEP 3333 (2003) | PEP 0 (2019) |
| **Request model** | Synchronous — one thread/process per request | Asynchronous — single event loop handles many requests concurrently |
| **WebSocket support** | No | Yes |
| **HTTP/2** | No | Yes |
| **Frameworks** | Django, Flask, Bottle | FastAPI, Starlette, Django 3.1+ |
| **Servers** | Gunicorn, uWSGI | Uvicorn, Hypercorn, Daphne |

## 3.3 Request Lifecycle

```mermaid
flowchart TD
    Client --> Uvicorn["Uvicorn ASGI Server"]
    Uvicorn --> Starlette["Starlette Routing"]
    Starlette --> Middleware["Middleware Stack\n(auth, logging, CORS, tracing)"]
    Middleware --> DI["Dependency Injection\n(resolve dependencies)"]
    DI --> Validation["Pydantic Validation\n(parse path/query/body)"]
    Validation --> RouteHandler["Route Handler Function\nasync def create_user(...)"]
    RouteHandler --> ResponseModel["Response Model\n(serialize output)"]
    ResponseModel --> Client
    Validation -- "Validation Error" --> 422["422 Unprocessable Entity"]
```

## 3.4 Core FastAPI Features

### Routing and Path Parameters

```python
from fastapi import FastAPI, Path, Query, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

app = FastAPI(title="ML Platform API", version="2.0.0")

class ModelStatus(str, Enum):
    training = "training"
    deployed = "deployed"
    archived = "archived"

class ModelResponse(BaseModel):
    id: int
    name: str
    status: ModelStatus
    accuracy: float = Field(ge=0.0, le=1.0)

    class Config:
        from_attributes = True  # allow ORM objects

@app.get(
    "/models/{model_id}",
    response_model=ModelResponse,
    status_code=200,
    summary="Get model by ID",
    tags=["Models"],
)
async def get_model(
    model_id: int = Path(gt=0, description="Model ID"),
    include_metrics: bool = Query(default=False),
) -> ModelResponse:
    model = await fetch_model_from_db(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model
```

### Request Body and Validation

```python
from pydantic import BaseModel, Field, model_validator
from typing import Optional
import re

class CreateModelRequest(BaseModel):
    name: str = Field(min_length=3, max_length=100, pattern=r"^[a-z0-9\-]+$")
    description: Optional[str] = Field(default=None, max_length=500)
    base_model: str
    hyperparams: dict[str, float | int | str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_hyperparams(self) -> "CreateModelRequest":
        if "learning_rate" in self.hyperparams:
            lr = self.hyperparams["learning_rate"]
            if not (1e-6 <= float(lr) <= 1.0):
                raise ValueError("learning_rate must be between 1e-6 and 1.0")
        return self

@app.post("/models", response_model=ModelResponse, status_code=201)
async def create_model(request: CreateModelRequest) -> ModelResponse:
    model = await save_model_to_db(request)
    return model
```

### Dependency Injection

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Reusable dependency: validate JWT and return user."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await fetch_user(user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_role(*roles: str):
    """Factory: return dependency that checks if user has one of the given roles."""
    async def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return role_checker

# Usage — DI is composable
@app.delete("/models/{model_id}", status_code=204)
async def delete_model(
    model_id: int,
    current_user: dict = Depends(require_role("admin", "ml-engineer")),
):
    await delete_model_from_db(model_id)
```

### Database with Async SQLAlchemy

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from fastapi import Depends
from typing import AsyncGenerator

DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/mlplatform"

engine = create_async_engine(DATABASE_URL, pool_size=20, max_overflow=10, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class ModelORM(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True, index=True)
    status: Mapped[str]
    accuracy: Mapped[float]

# Dependency — yields session, commits on success, rolls back on error
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

@app.get("/models/{model_id}")
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)):
    model = await db.get(ModelORM, model_id)
    if not model:
        raise HTTPException(404, detail="Not found")
    return model
```

### Middleware

```python
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        start_time = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"
        return response

app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Background Tasks and Lifespan

```python
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks
import asyncio

# Lifespan: startup and shutdown logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await engine.connect()
    await redis_client.ping()
    print("All connections established")
    yield
    # Shutdown
    await engine.dispose()
    await redis_client.close()
    print("Connections closed gracefully")

app = FastAPI(lifespan=lifespan)

# Background tasks — fire and forget after response
async def send_training_notification(model_id: int, user_email: str):
    await send_email(user_email, f"Model {model_id} training started")

@app.post("/models/{model_id}/train", status_code=202)
async def start_training(
    model_id: int,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    job_id = await submit_training_job(model_id)
    background_tasks.add_task(send_training_notification, model_id, current_user["email"])
    return {"job_id": job_id, "status": "queued"}
```

### Custom Exception Handlers

```python
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

logger = logging.getLogger(__name__)

class DomainError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code

@app.exception_handler(DomainError)
async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "message": exc.message},
    )

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"field": ".".join(str(l) for l in err["loc"]), "message": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"errors": errors})

@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception", extra={"path": request.url.path})
    return JSONResponse(status_code=500, content={"error": "internal_error"})
```

## 3.5 Async Programming Deep Dive

### Sync vs Async — the mental model

```mermaid
flowchart TD
    subgraph sync [Synchronous - Threading Model]
        T1["Thread 1: Request A"] -- "Blocked on DB\n(idle CPU)" --> T1_wait["...waiting..."]
        T2["Thread 2: Request B"] -- "Blocked on API\n(idle CPU)" --> T2_wait["...waiting..."]
    end
    subgraph asyncio_model [Asynchronous - Event Loop Model]
        EL["Single Event Loop\n(single thread)"] --> RqA["Start Request A"]
        RqA -- "await DB query" --> RqB["Start Request B while A waits"]
        RqB -- "await API call" --> RqC["Start Request C while B waits"]
        RqC --> EL
    end
```

### Key Concepts

| Concept | Definition | FastAPI relevance |
|---------|-----------|------------------|
| **Coroutine** | Function that can pause and resume — defined with `async def` | Route handlers, dependencies |
| **Event Loop** | Scheduler that runs coroutines — `asyncio.get_event_loop()` | Uvicorn manages one per worker |
| **`await`** | Pause current coroutine, yield control back to event loop | `await db.query(...)`, `await redis.get(...)` |
| **`async def`** | Define a coroutine function | All async route handlers |
| **GIL** | CPython's Global Interpreter Lock — only one thread runs Python at a time | asyncio avoids GIL contention — one thread, no locking |
| **Concurrency** | Multiple tasks make progress simultaneously (interleaved) | asyncio: one thread handles many I/O-bound requests |
| **Parallelism** | Multiple tasks run at the exact same time (multiple CPUs) | `ProcessPoolExecutor` or multiple Uvicorn workers |

### When to use `async def` vs `def` in FastAPI

```python
# async def — for I/O-bound operations
@app.get("/users/{id}")
async def get_user(id: int):
    user = await db.fetch_one(...)   # non-blocking DB query
    return user

# def — for CPU-bound operations (FastAPI runs them in a thread pool)
@app.post("/models/evaluate")
def evaluate_model(payload: EvalRequest):
    result = heavy_sklearn_computation(payload.data)   # CPU-bound, blocks
    return result
# FastAPI detects sync def and runs it in threadpool executor automatically

# WRONG: blocking call inside async def — blocks the event loop!
@app.get("/bad")
async def bad_handler():
    import time
    time.sleep(5)    # BLOCKS the event loop — all other requests stall
    return {}
```

## 3.6 FastAPI Security

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone

security = HTTPBearer()

def create_access_token(subject: str, roles: list[str], expires_minutes: int = 15) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {
        "sub": subject,
        "roles": roles,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": str(uuid.uuid4()),  # unique token ID for revocation
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        # Check revocation list (Redis)
        jti = payload.get("jti")
        if await redis.exists(f"revoked:{jti}"):
            raise HTTPException(status_code=401, detail="Token revoked")
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))
```

### Rate Limiting with Redis

```python
import redis.asyncio as aioredis
from fastapi import Request

redis_client = aioredis.from_url("redis://localhost:6379")

async def rate_limit(request: Request, max_requests: int = 100, window_seconds: int = 60):
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"

    current = await redis_client.incr(key)
    if current == 1:
        await redis_client.expire(key, window_seconds)

    remaining = max(0, max_requests - current)
    if current > max_requests:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(window_seconds), "X-RateLimit-Remaining": "0"},
        )
    return remaining
```

## 3.7 FastAPI Observability

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

# Structured logging
log = structlog.get_logger()

# OpenTelemetry tracing
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317")))
trace.set_tracer_provider(provider)
FastAPIInstrumentor.instrument_app(app)  # auto-traces all routes

# Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Correlation ID in logs
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    with structlog.contextvars.bound_contextvars(
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
    ):
        response = await call_next(request)
        log.info("request_completed", status=response.status_code)
        return response
```

## 3.8 Testing FastAPI

```python
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_get_model_success(client, mock_db):
    mock_db.get.return_value = ModelORM(id=1, name="gpt-mini", status="deployed", accuracy=0.94)
    response = await client.get("/models/1", headers={"Authorization": "Bearer valid-token"})
    assert response.status_code == 200
    assert response.json()["name"] == "gpt-mini"

@pytest.mark.asyncio
async def test_get_model_not_found(client, mock_db):
    mock_db.get.return_value = None
    response = await client.get("/models/999", headers={"Authorization": "Bearer valid-token"})
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_create_model_validation_error(client):
    response = await client.post("/models", json={"name": "x"})  # too short
    assert response.status_code == 422
    errors = response.json()["errors"]
    assert any(e["field"] == "name" for e in errors)

# Override dependency for testing
from app.main import app, get_db, get_current_user

def override_get_db():
    yield mock_session

app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = lambda: {"id": 1, "role": "admin"}
```

## 3.9 FastAPI Interview Questions

**Beginner:**
1. What is FastAPI? How does it differ from Flask?
2. What is Pydantic used for in FastAPI?
3. What is the difference between a path parameter and a query parameter?
4. How do you return a 404 error in FastAPI?
5. What does `response_model` do?

**Intermediate:**
1. Explain FastAPI's dependency injection system. What makes it powerful?
2. What is the difference between `async def` and `def` route handlers in FastAPI?
3. How does FastAPI generate OpenAPI documentation automatically?
4. Explain ASGI vs WSGI and why FastAPI uses ASGI.
5. What happens when Pydantic validation fails? What status code is returned?

**Advanced:**
1. How do you handle database transactions correctly with FastAPI and SQLAlchemy async?
2. How would you implement a production-grade rate limiter in FastAPI?
3. Explain how lifespan events work and why they replaced `@app.on_event`.
4. How do you avoid blocking the event loop in FastAPI?
5. Design a multi-tenant FastAPI service where each tenant has isolated data.

**Staff Engineer:**
1. How would you design a FastAPI service to serve 100k requests/second?
2. How do you implement distributed tracing across 10 FastAPI microservices?
3. Design a circuit breaker pattern for external API calls in FastAPI.
4. How do you handle zero-downtime deployments for a FastAPI service?
5. How would you implement an idempotent payment API in FastAPI?

---

# 4. gRPC Complete Guide

## 4.1 What is gRPC?

gRPC (gRPC Remote Procedure Call) is an open-source RPC framework developed by Google. It uses:
- **Protocol Buffers (protobuf)** for serialization — binary, compact, fast
- **HTTP/2** for transport — multiplexing, bidirectional streaming, header compression
- **Code generation** — `.proto` file → client stubs in any language

```mermaid
flowchart LR
    ProtoFile[".proto file\n(service definition)"] --> Compiler["protoc compiler"]
    Compiler --> ClientStub["Generated Client Stub\n(Python / Go / Java / ...)"]
    Compiler --> ServerBase["Generated Server Base\n(implement methods)"]
    ClientStub -- "Binary payload over HTTP/2" --> ServerBase
```

### Why gRPC is Faster than REST

| Dimension | REST + JSON | gRPC + Protobuf |
|-----------|------------|-----------------|
| **Serialization** | Text → parse JSON → allocate objects | Binary → direct memory mapping |
| **Payload size** | JSON: verbose field names, text encoding | Protobuf: field numbers, varint encoding — 3-10× smaller |
| **Transport** | HTTP/1.1: one request per connection | HTTP/2: many streams on one connection |
| **Code gen** | Optional (OpenAPI codegen) | Mandatory — always up to date |
| **Contract** | Optional (OpenAPI schema) | Mandatory — `.proto` is the source of truth |
| **Streaming** | Limited (SSE for server streaming) | First-class bidirectional streaming |

## 4.2 Protocol Buffers Deep Dive

### .proto Syntax

```protobuf
syntax = "proto3";

package mlplatform.v1;

import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";

// Service definition
service ModelService {
  rpc GetModel (GetModelRequest) returns (ModelResponse);
  rpc CreateModel (CreateModelRequest) returns (ModelResponse);
  rpc WatchTraining (WatchTrainingRequest) returns (stream TrainingEvent);
  rpc BatchPredict (stream PredictRequest) returns (stream PredictResponse);
  rpc DeleteModel (DeleteModelRequest) returns (google.protobuf.Empty);
}

// Message definitions
message ModelResponse {
  int64 id = 1;
  string name = 2;
  ModelStatus status = 3;
  double accuracy = 4;
  google.protobuf.Timestamp created_at = 5;
  map<string, string> tags = 6;
  repeated string supported_languages = 7;

  // oneof: only one of these fields will be set
  oneof deployment_config {
    GPUConfig gpu_config = 8;
    CPUConfig cpu_config = 9;
  }
}

enum ModelStatus {
  MODEL_STATUS_UNSPECIFIED = 0;  // proto3: always start enum at 0
  MODEL_STATUS_TRAINING = 1;
  MODEL_STATUS_DEPLOYED = 2;
  MODEL_STATUS_ARCHIVED = 3;
}

message GetModelRequest {
  int64 model_id = 1;
  optional bool include_metrics = 2;
}

message CreateModelRequest {
  string name = 1;
  string base_model = 2;
  map<string, double> hyperparams = 3;
  repeated string tags = 4;
}

message TrainingEvent {
  int64 model_id = 1;
  string event_type = 2;   // "epoch_complete", "error", "done"
  double loss = 3;
  double accuracy = 4;
  int32 epoch = 5;
}

message GPUConfig {
  int32 gpu_count = 1;
  string gpu_type = 2;
}

message CPUConfig {
  int32 cpu_cores = 1;
  string instance_type = 2;
}

// Reserved field numbers/names to prevent accidental reuse
message DeleteModelRequest {
  int64 model_id = 1;
  reserved 2, 3;
  reserved "force_delete";
}
```

### Protobuf Wire Format and Field Numbers

Field numbers (1, 2, 3…) — not names — are encoded in the binary format. This is why:
- **Never reuse** field numbers (even after deletion) — it breaks deserialization
- **Never change** field numbers — old clients will misinterpret fields
- Use `reserved` to permanently block reuse of old numbers/names

### Backward and Forward Compatibility

| Change | Safe? | Reason |
|--------|-------|--------|
| Add new field (with default) | Yes | Old clients ignore unknown fields |
| Remove a field + reserve its number | Yes | Old clients won't see the field |
| Rename a field | Yes | Wire format uses number, not name |
| Change field number | **No** | Breaks all existing clients |
| Change field type incompatibly | **No** | Binary decode will corrupt data |
| Change from `repeated` to singular | **No** | Multiple values will be dropped |

## 4.3 gRPC Communication Types

### Type 1: Unary RPC

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: GetModel(GetModelRequest)
    Note over S: Process request
    S->>C: ModelResponse
```

**Use case:** Standard request-response. CRUD operations.

```python
# Python gRPC server
import grpc
from concurrent import futures
import mlplatform_pb2
import mlplatform_pb2_grpc

class ModelServicer(mlplatform_pb2_grpc.ModelServiceServicer):
    async def GetModel(self, request, context):
        model = await db.get_model(request.model_id)
        if not model:
            context.abort(grpc.StatusCode.NOT_FOUND, f"Model {request.model_id} not found")
        return mlplatform_pb2.ModelResponse(
            id=model.id,
            name=model.name,
            status=mlplatform_pb2.MODEL_STATUS_DEPLOYED,
            accuracy=model.accuracy,
        )

async def serve():
    server = grpc.aio.server()
    mlplatform_pb2_grpc.add_ModelServiceServicer_to_server(ModelServicer(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    await server.wait_for_termination()

# Python gRPC client
async def get_model_client(model_id: int):
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = mlplatform_pb2_grpc.ModelServiceStub(channel)
        response = await stub.GetModel(
            mlplatform_pb2.GetModelRequest(model_id=model_id),
            timeout=5.0,   # deadline
        )
        return response
```

```go
// Go gRPC server
package main

import (
    "context"
    "google.golang.org/grpc"
    pb "github.com/example/mlplatform/gen"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

type modelServer struct {
    pb.UnimplementedModelServiceServer
    db Database
}

func (s *modelServer) GetModel(ctx context.Context, req *pb.GetModelRequest) (*pb.ModelResponse, error) {
    model, err := s.db.GetModel(ctx, req.ModelId)
    if err != nil {
        return nil, status.Errorf(codes.NotFound, "model %d not found", req.ModelId)
    }
    return &pb.ModelResponse{
        Id:       model.ID,
        Name:     model.Name,
        Accuracy: model.Accuracy,
    }, nil
}

func main() {
    lis, _ := net.Listen("tcp", ":50051")
    s := grpc.NewServer(
        grpc.UnaryInterceptor(loggingInterceptor),
        grpc.UnaryInterceptor(authInterceptor),
    )
    pb.RegisterModelServiceServer(s, &modelServer{})
    s.Serve(lis)
}
```

### Type 2: Server Streaming RPC

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: WatchTraining(WatchTrainingRequest)
    S-->>C: TrainingEvent{epoch=1, loss=2.3}
    S-->>C: TrainingEvent{epoch=2, loss=1.8}
    S-->>C: TrainingEvent{epoch=3, loss=1.2}
    S-->>C: TrainingEvent{event_type="done"}
    Note over S: Stream closed
```

**Use cases:** Training progress streaming, log tailing, real-time model metrics, LLM token streaming.

```python
class ModelServicer(mlplatform_pb2_grpc.ModelServiceServicer):
    async def WatchTraining(self, request, context):
        """Server streams training events to client."""
        model_id = request.model_id
        async for event in get_training_event_stream(model_id):
            if context.cancelled():
                break
            yield mlplatform_pb2.TrainingEvent(
                model_id=model_id,
                event_type=event["type"],
                loss=event["loss"],
                epoch=event["epoch"],
            )

# Client consuming the stream
async def watch_training(model_id: int):
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = mlplatform_pb2_grpc.ModelServiceStub(channel)
        async for event in stub.WatchTraining(
            mlplatform_pb2.WatchTrainingRequest(model_id=model_id)
        ):
            print(f"Epoch {event.epoch}: loss={event.loss:.4f}")
            if event.event_type == "done":
                break
```

### Type 3: Client Streaming RPC

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C-->>S: PredictRequest{chunk=1}
    C-->>S: PredictRequest{chunk=2}
    C-->>S: PredictRequest{chunk=3}
    Note over C: Stream closed
    S->>C: PredictResponse{aggregated result}
```

**Use cases:** Upload large files for batch prediction, streaming sensor data for real-time model input.

```python
class ModelServicer(mlplatform_pb2_grpc.ModelServiceServicer):
    async def BatchPredict(self, request_iterator, context):
        """Client streams requests; server returns one aggregated response."""
        all_inputs = []
        async for request in request_iterator:
            all_inputs.append(request.input_data)
        predictions = await run_batch_inference(all_inputs)
        return mlplatform_pb2.BatchPredictResponse(predictions=predictions)
```

### Type 4: Bidirectional Streaming RPC

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C-->>S: Message chunk 1
    S-->>C: Acknowledgment / partial result
    C-->>S: Message chunk 2
    S-->>C: Streaming output token 1
    S-->>C: Streaming output token 2
    Note over C,S: Both sides send independently
```

**Use cases:** Real-time chat with LLM, interactive voice agents, trading order book, collaborative editing.

```python
class LLMServicer:
    async def StreamChat(self, request_iterator, context):
        """Full bidirectional: client streams tokens in, server streams tokens out."""
        async for user_message in request_iterator:
            async for token in llm.generate_stream(user_message.content):
                yield mlplatform_pb2.ChatResponse(token=token)
```

## 4.4 HTTP/2 Deep Dive

```mermaid
flowchart TD
    subgraph http1 [HTTP/1.1 - One Request Per Connection]
        Conn1["Connection 1"] --> Req1["Request 1"] --> Resp1["Response 1"]
        Conn2["Connection 2"] --> Req2["Request 2"] --> Resp2["Response 2"]
        Conn3["Connection 3"] --> Req3["Request 3"] --> Resp3["Response 3"]
    end
    subgraph http2 [HTTP/2 - Multiple Streams Per Connection]
        SingleConn["Single TCP Connection"] --> Stream1["Stream 1: Request A"]
        SingleConn --> Stream2["Stream 2: Request B"]
        SingleConn --> Stream3["Stream 3: Request C"]
        Stream1 --> RespA["Response A"]
        Stream2 --> RespB["Response B"]
        Stream3 --> RespC["Response C"]
    end
```

| Feature | HTTP/1.1 | HTTP/2 |
|---------|---------|--------|
| **Multiplexing** | One request per connection (head-of-line blocking) | Multiple streams on single connection |
| **Header compression** | Plaintext headers, repeated on every request | HPACK compression — headers encoded and cached |
| **Streaming** | Not native | Native streams with flow control |
| **Server Push** | Not supported | Server can push resources proactively |
| **Binary framing** | Text protocol | Binary frames — more efficient to parse |
| **Connection reuse** | Keep-Alive (limited) | Single long-lived connection |

**Streams and Frames:**
- A **connection** is one TCP socket
- A **stream** is a virtual channel within a connection (each RPC call = one stream)
- A **frame** is the atomic unit of HTTP/2 data (HEADERS frame, DATA frame, SETTINGS frame)
- **Flow control** prevents fast sender from overwhelming slow receiver

## 4.5 gRPC Security

```python
import grpc
import ssl

# TLS (server only authenticates)
credentials = grpc.ssl_channel_credentials(
    root_certificates=open("ca.crt", "rb").read()
)
channel = grpc.aio.secure_channel("api.example.com:443", credentials)

# mTLS (both sides authenticate)
credentials = grpc.ssl_channel_credentials(
    root_certificates=open("ca.crt", "rb").read(),
    private_key=open("client.key", "rb").read(),
    certificate_chain=open("client.crt", "rb").read(),
)

# Authentication interceptor
class JWTInterceptor(grpc.aio.ServerInterceptor):
    async def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata)
        token = metadata.get("authorization", "").replace("Bearer ", "")
        try:
            payload = jwt.decode(token, SECRET, algorithms=["HS256"])
            return await continuation(handler_call_details)
        except jwt.JWTError:
            async def abort(request, context):
                await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token")
            return grpc.unary_unary_rpc_method_handler(abort)
```

## 4.6 gRPC in Microservices

### Deadlines, Timeouts, and Cancellation

```python
# Set deadline per-call — propagate through call chain
try:
    response = await stub.GetModel(
        request,
        timeout=2.0,      # 2 second deadline
        metadata=[("x-request-id", request_id)],
    )
except grpc.aio.AioRpcError as e:
    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        # Log, increment metric, return cached value or error
        metrics.increment("grpc.deadline_exceeded", tags=["service:model"])
    elif e.code() == grpc.StatusCode.UNAVAILABLE:
        # Retry with exponential backoff
        ...
```

### gRPC Health Checking

```python
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

health_servicer = health.HealthServicer()

async def serve():
    server = grpc.aio.server()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    # Set status per service
    health_servicer.set("mlplatform.v1.ModelService", health_pb2.HealthCheckResponse.SERVING)
```

### Interceptors (Middleware equivalent)

```python
class MetricsInterceptor(grpc.aio.ServerInterceptor):
    async def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method
        start = time.monotonic()
        handler = await continuation(handler_call_details)

        async def wrapper(request, context):
            try:
                response = await handler.unary_unary(request, context)
                metrics.increment(f"grpc.success", tags=[f"method:{method}"])
                return response
            except Exception as e:
                metrics.increment(f"grpc.error", tags=[f"method:{method}"])
                raise
            finally:
                latency = time.monotonic() - start
                metrics.histogram("grpc.latency", latency, tags=[f"method:{method}"])

        return grpc.unary_unary_rpc_method_handler(wrapper)
```

## 4.7 gRPC Interview Questions

**Beginner:**
1. What is gRPC and how does it differ from REST?
2. What is Protocol Buffers? Why use it instead of JSON?
3. What are the four types of gRPC communication?
4. What is HTTP/2 and why does gRPC use it?

**Intermediate:**
1. How do you handle backward compatibility in protobuf?
2. What is a gRPC deadline and why is it important to set one?
3. How do gRPC interceptors work? Give an example.
4. How does multiplexing in HTTP/2 improve performance over HTTP/1.1?

**Advanced:**
1. How would you implement gRPC streaming for LLM token generation?
2. How do you implement circuit breaking in gRPC?
3. How does gRPC load balancing work in Kubernetes?
4. Explain gRPC flow control and backpressure.

**Principal Engineer:**
1. Design a gRPC service mesh for 50 microservices with mTLS and distributed tracing.
2. How would you migrate a large REST API fleet to gRPC with zero downtime?
3. Design a gRPC gateway that translates REST→gRPC for browser clients.
4. How would you handle gRPC streaming for a 1M concurrent users LLM platform?

---

# 5. REST vs FastAPI vs gRPC

## Comprehensive Decision Matrix

| Dimension | REST (JSON/HTTP) | FastAPI | gRPC (Protobuf/HTTP2) |
|-----------|-----------------|---------|----------------------|
| **Protocol** | HTTP/1.1 or 2 | HTTP/1.1 or 2 (ASGI) | HTTP/2 only |
| **Data format** | JSON (text) | JSON via Pydantic | Protobuf (binary) |
| **Payload size** | Large (verbose JSON) | Large (JSON) | 3–10× smaller (binary) |
| **Latency** | Higher (text parse) | Medium | Lowest (binary, no parse) |
| **Throughput** | Medium | High (async) | Highest |
| **Streaming** | SSE / WebSocket (manual) | SSE / WebSocket | Native (4 types) |
| **Code generation** | Optional (OpenAPI) | Auto (OpenAPI+Pydantic) | Mandatory (protoc) |
| **Contract** | Optional (OpenAPI) | Auto-generated | Mandatory (.proto) |
| **Browser support** | Full | Full | Partial (grpc-web) |
| **Human readable** | Yes | Yes | No (binary) |
| **Debugging** | curl, Postman, browser | Swagger UI | grpcurl, Postman |
| **Learning curve** | Low | Low | Medium |
| **Type safety** | Runtime (no codegen) | Runtime (Pydantic) | Compile-time (generated) |
| **Versioning** | URI / Header | URI / Header | Package versioning in .proto |
| **Security** | HTTPS + JWT/OAuth | HTTPS + JWT/OAuth | TLS/mTLS built-in |
| **Load balancing** | Standard (L4/L7) | Standard | L4 (needs L7 for streaming) |
| **API gateway** | Full support | Full support | Limited (needs transcoding) |
| **Internal services** | Good | Good | Best |
| **External APIs** | Best | Best | Not recommended |
| **ML model serving** | Good | Best (Python ecosystem) | Good (cross-language) |
| **LLM serving** | Good (SSE) | Best (native SSE/WS) | Good (server streaming) |
| **MLOps pipelines** | Good | Best | Good |
| **Multi-language services** | Medium | Python only | Best (any language) |

## When to Use Which

```mermaid
flowchart TD
    start[New API Requirement] --> external{External or internal?}
    external -- External / browser clients --> rest[REST + JSON\nFastAPI or similar]
    external -- Internal service-to-service --> scale{High throughput\n> 10k rps?}
    scale -- No --> fastapi[FastAPI\nFast development, Python ecosystem]
    scale -- Yes --> multilang{Multiple languages?}
    multilang -- No --> fastapi
    multilang -- Yes --> grpc[gRPC\nBest performance + cross-language]
    fastapi -- LLM serving --> fastapi
    grpc -- Feature store / inference pipeline --> grpc
```

---

# 6. System Design Perspective

## 6.1 API Gateway

An **API Gateway** is the single entry point for all external traffic. It handles:
- Authentication and authorization
- Rate limiting and throttling
- Request routing to downstream services
- Request/response transformation
- SSL termination
- Observability (logging, metrics, tracing)

```mermaid
flowchart LR
    Internet --> APIGW["API Gateway\n(Kong / AWS API GW / Envoy)"]
    APIGW -- "Auth + Rate Limit" --> UserSvc["User Service"]
    APIGW -- "Auth + Rate Limit" --> ModelSvc["Model Service"]
    APIGW -- "Auth + Rate Limit" --> InferenceSvc["Inference Service"]
    APIGW -- "REST→gRPC transcoding" --> FeatureSvc["Feature Store\n(gRPC internal)"]
```

**Popular API Gateways:**

| Product | Best for | Notes |
|---------|---------|-------|
| Kong | Self-hosted, plugin ecosystem | Lua plugins; enterprise features |
| AWS API Gateway | AWS-native, serverless | Lambda integration; pay-per-request |
| Envoy | Service mesh, high-perf | Xds control plane; used in Istio |
| Nginx | Reverse proxy, simple routing | Fast, minimal feature set |
| Traefik | Kubernetes-native | Auto-discovers services via labels |

## 6.2 Load Balancer

```mermaid
flowchart LR
    Client --> LB["Load Balancer\n(L4/L7)"]
    LB -- "Round Robin / Least Conn" --> Inst1["Instance 1"]
    LB -- "Round Robin / Least Conn" --> Inst2["Instance 2"]
    LB -- "Round Robin / Least Conn" --> Inst3["Instance 3"]
    Inst1 --> DB[("Database\nwith Read Replicas")]
    Inst2 --> DB
    Inst3 --> DB
```

| Strategy | How it works | Best for |
|----------|-------------|---------|
| Round Robin | Distribute requests sequentially | Homogeneous workloads |
| Least Connections | Send to server with fewest active connections | Variable request durations |
| IP Hash | Hash client IP → same server | Sticky sessions (avoid if possible) |
| Weighted | Assign weights; route proportionally | Different instance capacities |
| Health-based | Only route to passing health check instances | Always enabled |

## 6.3 Caching Architecture

```mermaid
flowchart LR
    Client --> CDN["CDN\n(static assets, public responses)"]
    CDN -- Cache Miss --> APIGW
    APIGW --> AppServer["App Server"]
    AppServer --> Redis["Redis Cache\n(L1 cache: hot data)"]
    Redis -- Cache Miss --> DB[("Database\n(L2: source of truth)")]
```

**Cache invalidation strategies:**
- **TTL (Time-to-Live):** Expire entry after N seconds. Simple but can serve stale data.
- **Cache-Aside (Lazy Loading):** Read from cache; on miss, load from DB and populate cache.
- **Write-Through:** Write to cache and DB simultaneously. Always consistent; extra write latency.
- **Write-Behind (Write-Back):** Write to cache, async write to DB. Fast writes; risk of data loss.
- **Event-driven invalidation:** On DB update, publish event → cache invalidation worker clears key.

## 6.4 Message Queues

```mermaid
flowchart LR
    Producer["Model Training\nService"] -- "Publish: TrainingComplete" --> Kafka["Kafka Topic:\nml.events"]
    Kafka -- "Subscribe" --> NotifConsumer["Notification Consumer\n(send email)"]
    Kafka -- "Subscribe" --> MetricsConsumer["Metrics Consumer\n(update dashboard)"]
    Kafka -- "Subscribe" --> ModelRegistryConsumer["Model Registry\n(index new model)"]
```

| System | Best for | Delivery | Ordering |
|--------|---------|---------|---------|
| **Kafka** | High-throughput streaming, event sourcing, audit log | At-least-once; exactly-once (with transactions) | Partition-level ordering |
| **RabbitMQ** | Task queues, RPC-over-queue, complex routing | At-least-once | Queue-level ordering |
| **NATS** | Low-latency pub/sub, IoT, ephemeral streams | At-most-once (core) / at-least-once (JetStream) | Subject-level |
| **Redis Streams** | Lightweight streaming, same Redis instance | At-least-once | Append-only |

---

# 7. AI Engineering Perspective

## 7.1 Model Serving Architectures

```mermaid
flowchart TD
    Client --> Gateway["Model Gateway\n(FastAPI / Triton / vLLM)"]
    Gateway --> Router["Request Router\n(route by model, version, load)"]
    Router --> OnlineInference["Online Inference\n(< 100ms SLA)\nFastAPI + Torch"]
    Router --> BatchInference["Batch Inference\n(async, throughput optimized)\nRay / Celery"]
    Router --> StreamInference["Streaming Inference\n(token streaming)\nFastAPI SSE / gRPC"]
    OnlineInference --> GPUCluster["GPU Cluster\n(Model Replicas)"]
    BatchInference --> GPUCluster
    StreamInference --> GPUCluster
    GPUCluster --> FeatureStore["Feature Store\n(Feast / Tecton)"]
    GPUCluster --> VectorDB["Vector DB\n(Qdrant / pgvector)"]
```

## 7.2 LLM Serving with FastAPI — Streaming

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

async def token_generator(prompt: str):
    """Stream tokens from LLM as Server-Sent Events."""
    async for token in llm.generate_stream(prompt):
        # SSE format: "data: {json}\n\n"
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0)   # yield control to event loop
    yield "data: [DONE]\n\n"

@app.post("/v1/completions/stream")
async def stream_completion(request: CompletionRequest):
    return StreamingResponse(
        token_generator(request.prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # disable Nginx buffering
        },
    )
```

## 7.3 RAG System API Design

```mermaid
flowchart LR
    Query["User Query"] --> EmbedSvc["Embedding Service\n(FastAPI + text-embedding-3)"]
    EmbedSvc --> VectorSearch["Vector Search\n(Qdrant client)"]
    VectorSearch --> Docs["Retrieved Docs\n(top-k)"]
    Docs --> Reranker["Reranker\n(Cohere / local model)"]
    Reranker --> ContextBuilder["Context Builder"]
    Query --> ContextBuilder
    ContextBuilder --> LLM["LLM Service\n(FastAPI + vLLM)"]
    LLM --> StreamResponse["Streaming Response"]
```

```python
@app.post("/v1/rag/query")
async def rag_query(request: RAGRequest) -> StreamingResponse:
    # Step 1: Embed the query
    query_embedding = await embed_service.embed(request.query)

    # Step 2: Retrieve relevant chunks
    chunks = await qdrant_client.search(
        collection_name="knowledge_base",
        query_vector=query_embedding,
        limit=10,
        with_payload=True,
    )

    # Step 3: Rerank
    reranked = await reranker.rerank(request.query, chunks, top_n=3)

    # Step 4: Build context
    context = "\n\n".join(chunk.payload["text"] for chunk in reranked)

    # Step 5: Stream LLM response
    return StreamingResponse(
        llm_stream(context=context, query=request.query),
        media_type="text/event-stream",
    )
```

## 7.4 MCP Servers

**Model Context Protocol (MCP)** servers expose tools and resources that AI agents can call. They communicate over stdio or HTTP/SSE.

```python
# FastAPI as an MCP server
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ML Platform MCP Server")

@mcp.tool()
async def search_models(query: str, status: str = "deployed") -> list[dict]:
    """Search for deployed models by query."""
    return await model_db.search(query=query, status=status)

@mcp.tool()
async def run_inference(model_id: int, inputs: list[float]) -> dict:
    """Run inference on a deployed model."""
    return await inference_service.predict(model_id, inputs)

@mcp.resource("models://list")
async def list_models() -> str:
    """Return all deployed models as a resource."""
    models = await model_db.list_deployed()
    return json.dumps([m.dict() for m in models])
```

## 7.5 Feature Store API

```python
# Feature Store REST API — FastAPI
@app.get("/features/{entity_id}", response_model=FeatureVector)
async def get_features(
    entity_id: str,
    feature_set: str = Query(..., description="Feature set name"),
    max_age_seconds: int = Query(default=300),
):
    """Get online features for real-time inference."""
    features = await feature_store.get_online_features(
        entity_id=entity_id,
        feature_set=feature_set,
        max_staleness=timedelta(seconds=max_age_seconds),
    )
    if not features:
        raise HTTPException(404, "No features found for this entity")
    return features

# gRPC for batch feature retrieval — performance critical
# See proto definition in Section 4
```

---

# 8. Real Production Case Studies

## 8.1 OpenAI-like Platform Architecture

```mermaid
flowchart TD
    SDK["Client SDKs\n(Python/JS/Go)"] --> APIGW["API Gateway\n(Rate limit, auth, routing)"]
    APIGW --> ChatAPI["Chat Completions\n(FastAPI + async)"]
    APIGW --> EmbedAPI["Embeddings\n(FastAPI + async)"]
    APIGW --> ImageAPI["Images\n(FastAPI + async)"]
    ChatAPI --> LoadBalancer["GPU Load Balancer\n(consistent hashing by model)"]
    LoadBalancer --> vLLM1["vLLM Instance 1\n(GPT-4o)"]
    LoadBalancer --> vLLM2["vLLM Instance 2\n(GPT-4o-mini)"]
    ChatAPI --> Redis[("Redis\nRate limit counters\nIdempotency keys")]
    ChatAPI --> PG[("PostgreSQL\nRequest logs\nUsage billing")]
    ChatAPI --> Kafka["Kafka\nAudit log\nUsage events"]
```

**Key design decisions:**
- **Streaming by default:** All completions return SSE stream; non-streaming clients buffer internally
- **Idempotency keys:** Prevents double-billing on retry
- **Rate limiting per API key** at gateway, Redis sliding window
- **Usage metered in Kafka** → async billing pipeline
- **Model routing** by `model` field in request → consistent hash to specialized GPU fleet

## 8.2 ML Platform Architecture

```mermaid
flowchart LR
    DataScientist["Data Scientist\n(Jupyter / SDK)"] --> ExperimentAPI["Experiment API\n(FastAPI)"]
    ExperimentAPI --> TrainingOrch["Training Orchestrator\n(Celery / Ray)"]
    TrainingOrch --> GPU["GPU Training Cluster\n(PyTorch DDP / FSDP)"]
    GPU --> ModelRegistry["Model Registry\n(MLflow + S3)"]
    ModelRegistry --> DeployAPI["Deploy API\n(FastAPI)"]
    DeployAPI --> KubeOperator["Kubernetes Operator\n(Seldon / KServe)"]
    KubeOperator --> ServingPod["Model Serving Pod\n(gRPC + Triton)"]
    ServingPod --> FeatureStore["Online Feature Store\n(Redis + gRPC)"]
    ServingPod --> MonitorAPI["Monitoring API\n(FastAPI + Prometheus)"]
```

**Component responsibilities:**

| Component | Protocol | Why |
|-----------|---------|-----|
| Experiment API | REST (FastAPI) | External, used by humans |
| Training Orchestrator | REST + Kafka | Async job dispatch |
| Model Registry | REST | Standard MLflow HTTP API |
| Feature Store | gRPC | Low-latency, high-throughput, internal |
| Model Serving | gRPC | Maximum throughput for inference |
| Monitoring API | REST + Prometheus metrics endpoint | External dashboard, standard protocol |

---

# 9. Failure Scenarios

## 9.1 Failure Taxonomy and Remediation

| Failure | Symptom | Diagnosis | Remediation |
|---------|---------|-----------|------------|
| **Timeout** | Requests hang, 504 upstream | Check slow query logs, trace spans | Add deadlines; async processing for slow ops |
| **Retry storm** | 503 cascade during recovery | Spike in error rate immediately before recovery | Exponential backoff + jitter; circuit breaker |
| **Memory leak** | Gradual OOM, pod restarts | `RSS` growing monotonically in metrics | Profile with `memory_profiler`; fix resource leaks; connection not closed |
| **Connection leak** | `Too many open files` error | `ss -s` shows many CLOSE_WAIT or TIME_WAIT | Use connection pooling; ensure contexts/sessions are closed |
| **N+1 queries** | High DB CPU, low app CPU | Slow query log; trace shows 100 tiny queries | Eager loading; DataLoader; denormalization |
| **Thundering herd** | DB spike when cache expires | Correlation: cache TTL aligns with DB spike | Jitter on TTL; cache warming; probabilistic early refresh |
| **Deadlock** | Requests freeze, manual rollback needed | DB deadlock logs | Consistent lock ordering; short transactions; use `SELECT FOR UPDATE SKIP LOCKED` |
| **DNS failure** | All downstream calls fail | DNS resolution errors in traces | DNS caching (TTL > 0); fallback to IP; retry on NXDOMAIN |
| **Certificate expiry** | All TLS connections fail | `certificate has expired` in logs | Automate renewal (cert-manager); alert at 30/7/1 day before expiry |
| **Backpressure ignored** | Memory grows unbounded, OOM | Queue depth growing in metrics | Implement queue depth limits; drop or reject when full; flow control |
| **Rate limiting client** | 429s from external API | API error logs | Exponential backoff; request queue; token bucket client-side |
| **Partial failure in distributed tx** | Inconsistent data | Cross-service data divergence | Saga pattern; idempotent compensating transactions; outbox pattern |
| **Network partition** | Subset of services unreachable | Connectivity errors from specific nodes | Circuit breaker; fallback responses; health check routing |
| **High GC pause (Python)** | Periodic latency spikes | Latency histogram has long tail | Reduce object allocations; use `PyPy`; optimize hot paths |

## 9.2 Circuit Breaker Pattern

```mermaid
stateDiagram-v2
    Closed --> Open : Failure threshold exceeded
    Open --> HalfOpen : Timeout elapsed
    HalfOpen --> Closed : Test request succeeded
    HalfOpen --> Open : Test request failed
```

```python
import asyncio
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject all requests
    HALF_OPEN = "half_open" # Testing if upstream recovered

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30, success_threshold=2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at: datetime | None = None

    async def call(self, fn, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if datetime.now() - self.opened_at > timedelta(seconds=self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker OPEN — request rejected")

        try:
            result = await fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def _on_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = datetime.now()
```

## 9.3 Exponential Backoff with Jitter

```python
import asyncio
import random

async def retry_with_backoff(fn, max_attempts=5, base_delay=1.0, max_delay=60.0):
    """
    Full jitter: sleep = random(0, min(cap, base * 2^attempt))
    Prevents synchronized retries from all clients.
    """
    for attempt in range(max_attempts):
        try:
            return await fn()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            if attempt == max_attempts - 1:
                raise
            # Full jitter — randomize the entire sleep window
            cap = min(max_delay, base_delay * (2 ** attempt))
            sleep = random.uniform(0, cap)
            await asyncio.sleep(sleep)
```

---

# 10. Interview Crash Revision

## Top 100 REST Questions (with Answers)

### Fundamentals

| # | Question | Answer |
|---|----------|--------|
| 1 | What is REST? | Architectural style using HTTP, stateless client-server, uniform interface, cacheable |
| 2 | What are the 6 REST constraints? | Client-Server, Stateless, Cacheable, Uniform Interface, Layered System, Code-on-Demand |
| 3 | What is idempotency? | Same request N times = same result as 1 request. GET, PUT, DELETE are idempotent |
| 4 | What is safety? | Safe methods do not modify server state. GET, HEAD, OPTIONS are safe |
| 5 | Is PATCH idempotent? | Depends on implementation. JSON Patch is idempotent; custom semantics may not be |
| 6 | POST vs PUT vs PATCH? | POST=create, not idempotent; PUT=full replace, idempotent; PATCH=partial update |
| 7 | 401 vs 403? | 401=not authenticated (no valid credentials); 403=authenticated but not authorized |
| 8 | 200 vs 201 vs 202 vs 204? | 200=OK with body; 201=created; 202=async accepted; 204=success no body |
| 9 | 400 vs 422? | 400=malformed syntax; 422=syntax OK but semantically invalid (validation failed) |
| 10 | 502 vs 503 vs 504? | 502=bad gateway response; 503=service unavailable; 504=gateway timeout |

### URI and Resource Design

| # | Question | Answer |
|---|----------|--------|
| 11 | URI vs URL vs URN? | URI=identifier; URL=locatable URI; URN=permanent name URI |
| 12 | Noun vs verb in URIs? | Always nouns. Wrong: `/getUser`; Right: `GET /users/{id}` |
| 13 | Singular vs plural resources? | Always plural: `/users`, `/orders`, `/models` |
| 14 | Nested resource depth limit? | One level: `/users/{id}/orders`. Deeper = use query params |
| 15 | Filtering best practice? | Query params: `GET /orders?status=pending&created_after=2025-01-01` |
| 16 | Sorting convention? | `?sort=-created_at,amount` (minus = descending) |
| 17 | Pagination: offset vs cursor? | Cursor = stable under mutations, O(1) DB query. Offset = simple but drifts |
| 18 | Sparse fieldsets? | `?fields=id,name,email` — only return requested fields (saves bandwidth) |
| 19 | Sub-resource vs query param? | Sub-resource for relationships (`/users/1/orders`); query param for filtering |
| 20 | Search endpoint? | `GET /search?q=alice` or `/users?search=alice`. Never `POST /search` unless body needed |

### HTTP Methods

| # | Question | Answer |
|---|----------|--------|
| 21 | What does HEAD do? | Returns only headers of a GET response — no body. Used to check if resource exists |
| 22 | What does OPTIONS do? | Returns allowed methods. Used for CORS preflight |
| 23 | Can GET have a body? | Technically allowed by RFC but widely discouraged — use query params instead |
| 24 | Why DELETE usually returns 204? | No content to return after deletion. 200 with body is also acceptable |
| 25 | When to use PUT vs PATCH? | PUT if client has full resource state; PATCH if updating subset of fields |

### Authentication and Security

| # | Question | Answer |
|---|----------|--------|
| 26 | What is JWT? | JSON Web Token: base64(header).base64(payload).signature — stateless auth |
| 27 | JWT structure? | Header (alg/typ) + Payload (sub, exp, roles) + Signature (HMAC or RSA) |
| 28 | How to revoke a JWT? | Maintain JTI revocation list in Redis; short expiry (15 min) + refresh tokens |
| 29 | OAuth2 vs OpenID Connect? | OAuth2=authorization (access token); OIDC=authentication on top (ID token + userinfo) |
| 30 | OAuth2 grant types? | Authorization Code (user), Client Credentials (M2M), Device Code, Implicit (deprecated) |
| 31 | What is mTLS? | Both client and server present certificates for mutual authentication |
| 32 | API key vs JWT? | API key=static, no expiry, simple; JWT=self-contained, expiry, more secure |
| 33 | RBAC vs ABAC? | RBAC=roles; ABAC=evaluate policies on attributes (user+resource+environment) |
| 34 | What is CORS? | Browser security policy: server must permit cross-origin requests via `Access-Control-Allow-Origin` |
| 35 | What is CSRF? | Cross-Site Request Forgery — fix: CSRF tokens, `SameSite=Strict` cookies, check Origin header |

### Caching and Headers

| # | Question | Answer |
|---|----------|--------|
| 36 | ETag purpose? | Fingerprint of resource version; enables `304 Not Modified` on conditional GET |
| 37 | Cache-Control directives? | `max-age=N`, `no-cache` (validate), `no-store` (never cache), `private`, `public` |
| 38 | `no-cache` vs `no-store`? | `no-cache`=must revalidate; `no-store`=never cache at all (sensitive data) |
| 39 | Idempotency-Key purpose? | Client UUID on POST — server deduplicates, returns cached result on retry |
| 40 | X-Correlation-ID? | Unique ID carried across service hops for distributed tracing |
| 41 | If-Match vs If-None-Match? | `If-Match` for optimistic locking updates; `If-None-Match` for conditional GET |
| 42 | Retry-After header? | Number of seconds or HTTP date until client should retry (429, 503) |
| 43 | Strict-Transport-Security? | HSTS — browser always uses HTTPS for this domain; prevents downgrade attacks |
| 44 | `Location` header? | URL of newly created resource (201) or redirect target (301/302) |
| 45 | Content-Negotiation? | Client sends `Accept: application/json`; server returns matching format |

### API Design Patterns

| # | Question | Answer |
|---|----------|--------|
| 46 | What is the Richardson Maturity Model? | L0=POX, L1=Resources, L2=HTTP Verbs, L3=HATEOAS |
| 47 | What is HATEOAS? | Hypermedia As The Engine of App State — responses include links to next actions |
| 48 | Bulk API patterns? | `POST /users/bulk`, `PATCH /orders/bulk`. Return partial success with per-item status |
| 49 | Long-running operations? | Return 202 + job ID. Poll `GET /jobs/{id}` or use webhook callback |
| 50 | Webhook vs polling? | Webhook=server pushes (low latency, efficient); polling=client asks periodically (simple) |

### Versioning

| # | Question | Answer |
|---|----------|--------|
| 51 | URI versioning pros/cons? | Pro: visible, cacheable, easy to route; Con: pollutes URI |
| 52 | Header versioning pros/cons? | Pro: clean URI; Con: harder to test in browser |
| 53 | How to deprecate an API version? | Announce date, add `Deprecation` header, log traffic, sunset when zero traffic |
| 54 | Semantic versioning for APIs? | Major=breaking change; minor=backward-compatible addition; patch=bug fix |
| 55 | What makes a breaking change? | Remove field, rename field, change field type, change required/optional, change behavior |

### Performance and Reliability

| # | Question | Answer |
|---|----------|--------|
| 56 | Rate limiting algorithms? | Token Bucket (burst OK), Leaky Bucket (smooth), Fixed Window (simple), Sliding Window (accurate) |
| 57 | Thundering herd problem? | Cache expires → all clients hit DB. Fix: jitter TTL, probabilistic early refresh |
| 58 | Circuit breaker pattern? | CLOSED→OPEN on failures; HALF_OPEN after timeout to test recovery |
| 59 | Exponential backoff with jitter? | `sleep = random(0, min(cap, base * 2^attempt))` — prevents synchronized retries |
| 60 | Connection pooling purpose? | Reuse TCP connections to DB — avoid 3-way handshake + auth overhead per query |
| 61 | What causes N+1 queries? | Fetch 100 users → 100 separate queries for orders. Fix: eager loading |
| 62 | What is backpressure? | Downstream signal to upstream to slow down — prevents memory exhaustion |
| 63 | Partial failure in distributed tx? | Use Saga pattern: each step has compensating transaction; outbox pattern for reliability |
| 64 | Optimistic vs pessimistic locking? | Optimistic=ETag/version check on write (no lock); Pessimistic=`SELECT FOR UPDATE` (locks row) |
| 65 | What is the outbox pattern? | Write event to local DB table atomically with business data; worker publishes to queue |

### Production and Operations

| # | Question | Answer |
|---|----------|--------|
| 66 | Blue-green deployment? | Two identical environments; switch traffic instantly; immediate rollback capability |
| 67 | Canary deployment? | Route small % to new version; gradually increase if metrics healthy |
| 68 | How to do zero-downtime deploy? | Rolling update; graceful shutdown (drain connections, finish in-flight requests) |
| 69 | What is a health check endpoint? | `GET /health` returns 200 if service is ready; used by load balancer and K8s |
| 70 | Readiness vs liveness probe? | Readiness=ready to receive traffic; Liveness=process not dead (restart if fails) |
| 71 | API observability pillars? | Logs (what happened), Metrics (how much), Traces (where time was spent) |
| 72 | What is distributed tracing? | Follow a request across service hops via trace ID; see per-service latency |
| 73 | OpenTelemetry? | Vendor-neutral SDK for traces, metrics, logs. Export to Jaeger, Grafana, Datadog |
| 74 | What is a dead letter queue? | Messages that failed all retries go here for manual inspection |
| 75 | How to test for race conditions? | Load testing with concurrent requests; property-based testing; chaos engineering |

### Advanced and Architecture

| # | Question | Answer |
|---|----------|--------|
| 76 | What is an API gateway? | Single entry point: auth, rate limiting, routing, transformation, observability |
| 77 | Service mesh vs API gateway? | API GW=north-south traffic (external); Service Mesh=east-west traffic (internal) |
| 78 | What is service discovery? | Services register themselves; clients query registry to find endpoints (Consul, K8s DNS) |
| 79 | What is the BFF pattern? | Backend For Frontend — dedicated API layer per client type (mobile, web) |
| 80 | gRPC-gateway? | Generates REST proxy from proto annotations — allows browser clients to use REST while backend uses gRPC |
| 81 | What is OpenAPI 3.0? | Machine-readable API spec: endpoints, params, schemas, auth — generates docs + clients |
| 82 | Contract testing? | Verify consumer-provider contract without running full integration test (Pact) |
| 83 | API security scanning? | OWASP ZAP, Burp Suite — test for injection, XSS, SSRF, broken auth |
| 84 | What is SSRF? | Server-Side Request Forgery — attacker makes server call internal URLs |
| 85 | How to prevent SSRF? | Allowlist permitted domains; block 169.254.x.x (cloud metadata); validate URLs |
| 86 | What is request smuggling? | Discrepancy in HTTP parsing between gateway and backend; use consistent HTTP parsing |
| 87 | Explain the CAP theorem in API context | API availability vs consistency during network partition — choose based on business requirements |
| 88 | What is eventual consistency? | DB replicas may briefly diverge; reads may return stale data — acceptable for non-critical reads |
| 89 | How to handle large file uploads? | Multipart form or presigned S3 URL — direct upload from client to object store |
| 90 | Streaming large responses? | `StreamingResponse` (FastAPI) or chunked transfer encoding; avoid buffering entire response |
| 91 | What is request coalescing? | Multiple identical in-flight requests to cache → single backend call |
| 92 | What is singleflight? | Go pattern — duplicate in-flight requests share one result; prevents duplicate DB calls |
| 93 | API keys vs service accounts? | API keys=simple, static; Service accounts=RBAC, rotatable, audit trail |
| 94 | What is Mutual TLS (mTLS) vs OAuth2 mTLS? | Standard mTLS=cert authentication; OAuth2 mTLS token binding=cert bound to access token |
| 95 | How to trace a slow API request? | Distributed trace shows spans; find the longest span; check N+1, slow queries, lock waits |
| 96 | What is the strangler fig pattern? | Incrementally replace legacy API behind same URL; proxy routes new features to new service |
| 97 | What is API mocking? | Stub backend for frontend dev; contract tests; Wiremock, Mockoon, MSW |
| 98 | GraphQL vs REST? | GraphQL=flexible queries, one endpoint, overfetch/underfetch solved; REST=simple, cacheable, standards |
| 99 | When to use webhooks? | Push notifications: payment confirmation, ML training complete, CI/CD events |
| 100 | What is an API moat? | Business value locked in API ecosystem — developer adoption, SDKs, integrations |

## Top 100 FastAPI Questions (with Answers)

### Basics

| # | Question | Answer |
|---|----------|--------|
| 1 | What is FastAPI? | Modern Python web framework: ASGI, type hints, auto-validation, auto-docs |
| 2 | FastAPI vs Flask? | FastAPI: async, auto validation (Pydantic), auto docs, ASGI; Flask: sync, minimal, WSGI |
| 3 | FastAPI vs Django REST? | FastAPI: lightweight, auto docs, high perf; DRF: batteries-included, mature ecosystem |
| 4 | What is Pydantic? | Python data validation library — type hints become runtime validators |
| 5 | What is Starlette? | ASGI framework FastAPI is built on — handles routing, WebSocket, middleware |
| 6 | What is Uvicorn? | Fast ASGI server built on `uvloop` and `httptools` — serves FastAPI apps |
| 7 | ASGI vs WSGI? | ASGI: async, WebSocket, HTTP/2; WSGI: sync, one thread per request |
| 8 | What does `response_model` do? | Validate + serialize output; strip extra fields; document response schema |
| 9 | What HTTP status does Pydantic validation failure return? | 422 Unprocessable Entity |
| 10 | How to return custom status code? | `@app.post("/", status_code=201)` or `return Response(status_code=204)` |

### Routing and Parameters

| # | Question | Answer |
|---|----------|--------|
| 11 | How to declare a path parameter? | `@app.get("/users/{user_id}")` → `async def fn(user_id: int):` |
| 12 | How to add validation to path param? | `user_id: int = Path(gt=0, le=100000)` |
| 13 | How to declare a query parameter? | `async def fn(page: int = 1, limit: int = 20):` — no `Path`/`Body` means it's a query param |
| 14 | How to declare a required query param? | `async def fn(status: str):` — no default value = required |
| 15 | How to accept a list in query params? | `async def fn(tags: list[str] = Query(default=[]))` |
| 16 | Path param vs Query param vs Body? | Path=resource identifier; Query=filtering/pagination; Body=data being sent |
| 17 | How to make a route prefix? | `router = APIRouter(prefix="/api/v1", tags=["models"])` → `app.include_router(router)` |
| 18 | What are tags? | Group endpoints in OpenAPI docs — no functional impact |
| 19 | What is `response_model_exclude_unset`? | Only serialize fields that were explicitly set — avoids sending null defaults |
| 20 | How to deprecate a route? | `@app.get("/old", deprecated=True)` — shown as deprecated in Swagger UI |

### Dependency Injection

| # | Question | Answer |
|---|----------|--------|
| 21 | What is DI in FastAPI? | Functions that provide shared resources, declared with `Depends()` |
| 22 | How to create a reusable dependency? | `async def get_db(): yield session` → `db: Session = Depends(get_db)` |
| 23 | Are dependencies cached per request? | Yes — by default, `use_cache=True`; same dep called once per request |
| 24 | How to share DB transaction across deps? | Same session dep in all functions in the same request — cached |
| 25 | How to make dependencies composable? | Dep A can itself `Depends(dep_B)` — tree of dependencies |
| 26 | Security deps with `Depends`? | `current_user: dict = Depends(get_current_user)` — reuse auth in any route |
| 27 | `Security()` vs `Depends()`? | `Security()` adds to OpenAPI security schemes; `Depends()` is generic |
| 28 | Dependency with parameters? | Use a factory function: `def require_role(*roles): return inner_dep` |
| 29 | How to test with DI? | `app.dependency_overrides[real_dep] = mock_dep` — replace in tests |
| 30 | Class-based dependencies? | `class Pagination: def __init__(self, skip, limit)` → `Depends(Pagination)` |

### Pydantic and Validation

| # | Question | Answer |
|---|----------|--------|
| 31 | How to add field constraints? | `Field(min_length=3, max_length=100, ge=0, le=100, regex=r"^[a-z]+$")` |
| 32 | `model_validator` vs `field_validator`? | `field_validator` = validate one field; `model_validator` = validate after all fields parsed |
| 33 | How to allow extra fields? | `model_config = ConfigDict(extra='allow')` |
| 34 | How to forbid extra fields? | `model_config = ConfigDict(extra='forbid')` — returns 422 on extra fields |
| 35 | How to use ORM objects with Pydantic? | `model_config = ConfigDict(from_attributes=True)` → `MyModel.model_validate(orm_obj)` |
| 36 | Pydantic v1 vs v2? | v2: 10-50× faster (Rust core), `model_validate` replaces `from_orm`, `ConfigDict` replaces `Config` |
| 37 | How to exclude field from serialization? | `Field(exclude=True)` — field is validated but never serialized to JSON |
| 38 | Optional vs default None? | `Optional[str]` allows None; `str | None = None` is preferred in Python 3.10+ |
| 39 | How to validate on assignment? | `model_config = ConfigDict(validate_assignment=True)` |
| 40 | Custom serializer? | `@model_serializer` or `@field_serializer` decorators in Pydantic v2 |

### Async and Performance

| # | Question | Answer |
|---|----------|--------|
| 41 | `async def` vs `def` in route handlers? | `async def`=runs in event loop; `def`=FastAPI auto-runs in thread pool |
| 42 | When should you NOT use async def? | When calling blocking C-extension code that releases the GIL (numpy, torch) |
| 43 | How to run blocking code in async context? | `await asyncio.get_event_loop().run_in_executor(None, blocking_fn)` |
| 44 | What is the GIL? | CPython lock — only one thread runs Python bytecode at a time |
| 45 | How does asyncio avoid GIL issues? | Single thread, no locking needed — concurrency via coroutine switching |
| 46 | When does async NOT help? | CPU-bound work — use multiprocessing or async is fine since GIL released in C extensions |
| 47 | How many requests can Uvicorn handle? | Thousands concurrent (I/O bound); limited by event loop saturation on CPU-bound work |
| 48 | `uvicorn --workers N` vs `gunicorn -w N`? | Same effect — N worker processes; use N = 2×CPU+1 for general workloads |
| 49 | What is `asyncio.gather`? | Run multiple coroutines concurrently: `results = await asyncio.gather(coro1, coro2)` |
| 50 | What is a task vs coroutine? | Task=coroutine scheduled to run; `asyncio.create_task(coro)` — fire and forget |

### Middleware and Lifecycle

| # | Question | Answer |
|---|----------|--------|
| 51 | How to add middleware? | `app.add_middleware(MyMiddleware)` or `@app.middleware("http")` decorator |
| 52 | Middleware execution order? | Last-added is innermost; request: outer→inner; response: inner→outer |
| 53 | What is lifespan? | `@asynccontextmanager async def lifespan(app)` — code before yield=startup, after=shutdown |
| 54 | `@app.on_event` vs lifespan? | `on_event` is deprecated; use lifespan — single function for startup + shutdown |
| 55 | When to use background tasks? | Fire-and-forget after response: send email, emit metrics, log to external system |
| 56 | Background tasks vs Celery? | Background tasks = in-process after response; Celery = distributed worker queue |
| 57 | How to add CORS? | `app.add_middleware(CORSMiddleware, allow_origins=["..."], allow_methods=["*"])` |
| 58 | Gzip compression? | `app.add_middleware(GZipMiddleware, minimum_size=1000)` — compress responses > 1KB |
| 59 | How to log request/response? | `BaseHTTPMiddleware.dispatch` — inspect `request` before `call_next`, response after |
| 60 | Trusted Host middleware? | `TrustedHostMiddleware(allowed_hosts=["example.com"])` — reject requests with wrong Host |

### Database and State

| # | Question | Answer |
|---|----------|--------|
| 61 | Sync SQLAlchemy in FastAPI? | Works but blocks event loop — use `def` not `async def` so FastAPI uses threadpool |
| 62 | Async SQLAlchemy setup? | `create_async_engine(..., asyncpg)` + `AsyncSession` + `async_sessionmaker` |
| 63 | Transaction rollback on error? | `try: yield session; await session.commit() except: await session.rollback() raise` |
| 64 | N+1 with SQLAlchemy? | Use `selectinload(User.orders)` or `joinedload` in query — one query instead of N |
| 65 | Redis in FastAPI? | `redis.asyncio.from_url(...)` — async Redis client; share instance via lifespan |
| 66 | MongoDB in FastAPI? | `motor.motor_asyncio.AsyncIOMotorClient` — fully async MongoDB |
| 67 | How to run DB migrations? | Alembic — `alembic revision --autogenerate` + `alembic upgrade head` |
| 68 | What is connection pool exhaustion? | All DB connections in use; new requests wait; fix: increase pool, add queue depth alert |
| 69 | Alembic in production? | Run `alembic upgrade head` in container init; never auto-migrate in app code |
| 70 | How to seed test DB? | Pytest fixtures with `AsyncSession`, rollback after each test |

### Security

| # | Question | Answer |
|---|----------|--------|
| 71 | JWT validation in FastAPI? | `HTTPBearer` dependency → decode token → verify signature + expiry |
| 72 | OAuth2 password flow? | `OAuth2PasswordBearer(tokenUrl="/token")` — gets token from `Authorization: Bearer` header |
| 73 | How to store passwords? | `bcrypt` or `argon2` — never store plaintext or MD5/SHA1 hashes |
| 74 | HTTPS in FastAPI? | Terminate TLS at Nginx/load balancer; FastAPI gets plain HTTP internally |
| 75 | Rate limiting dependency? | Redis `INCR` + `EXPIRE` per IP/key; raise `HTTPException(429)` when exceeded |
| 76 | What is SSRF? | Attacker makes your server fetch internal URLs; validate all user-provided URLs |
| 77 | XSS prevention in API? | APIs return JSON not HTML; but validate output encoding and `Content-Type` |
| 78 | What is SQL injection? | User input in raw SQL; prevent with parameterized queries / SQLAlchemy ORM |
| 79 | Security headers? | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `HSTS` |
| 80 | Secrets management? | Never in code or env vars committed to git; use Vault, AWS Secrets Manager, K8s Secrets |

### Testing and Observability

| # | Question | Answer |
|---|----------|--------|
| 81 | How to test FastAPI? | `AsyncClient(transport=ASGITransport(app))` — no real server needed |
| 82 | How to mock external services? | `unittest.mock.patch` or `respx` for mocking HTTP clients |
| 83 | What is contract testing? | Verify API contract between consumer and provider without E2E test (Pact) |
| 84 | Load testing FastAPI? | `locust`, `k6`, or `wrk`; target p95/p99 latency under expected load |
| 85 | What is OpenTelemetry? | Vendor-neutral observability SDK — traces, metrics, logs; export to Jaeger, Grafana |
| 86 | Auto-instrumentation? | `FastAPIInstrumentor.instrument_app(app)` — traces all routes automatically |
| 87 | Prometheus metrics in FastAPI? | `prometheus_fastapi_instrumentator` — expose `/metrics` endpoint |
| 88 | Structured logging? | `structlog` or `python-json-logger` — JSON logs with correlation_id, user_id, etc. |
| 89 | Correlation ID propagation? | Middleware reads `X-Correlation-ID`, binds to log context, forwards in downstream calls |
| 90 | Profiling FastAPI? | `pyinstrument` or `cProfile` via middleware; `py-spy` for production profiling |

### Deployment and Advanced

| # | Question | Answer |
|---|----------|--------|
| 91 | Dockerfile for FastAPI? | Multi-stage build; `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4` |
| 92 | K8s deployment? | Deployment + Service + HPA (autoscale by CPU/request rate) |
| 93 | Graceful shutdown? | `SIGTERM` → Uvicorn stops accepting, drains in-flight requests → pod terminates |
| 94 | What is `uvloop`? | Drop-in replacement for `asyncio` event loop — 2-4× faster using libuv |
| 95 | Multiple apps on one Uvicorn? | `Starlette.Mount` or `fastapi.APIRouter` — all within one process |
| 96 | File uploads in FastAPI? | `UploadFile` type — async chunked reading; stream to S3 |
| 97 | WebSocket in FastAPI? | `@app.websocket("/ws")` + `await ws.accept()` + `await ws.receive_text()` |
| 98 | Server-Sent Events? | `StreamingResponse(generator, media_type="text/event-stream")` |
| 99 | How to version FastAPI APIs? | `app.include_router(v1_router, prefix="/v1")` and `app.include_router(v2_router, prefix="/v2")` |
| 100 | Horizontal scaling consideration? | Stateless design — no in-process state; use Redis for shared state; sticky sessions = bad |

## Top 100 gRPC Questions (with Answers)

### Fundamentals

| # | Question | Answer |
|---|----------|--------|
| 1 | What is gRPC? | Google's RPC framework: Protobuf serialization + HTTP/2 transport + code generation |
| 2 | gRPC vs REST? | gRPC: binary, streaming, codegen, faster; REST: JSON, human-readable, browser-native |
| 3 | What are Protocol Buffers? | Binary serialization with schema. Field numbers in wire format, not names |
| 4 | Why gRPC for internal services? | 3-10× smaller payload, multiplexing, streaming, type-safe codegen across languages |
| 5 | What is HTTP/2? | Binary multiplexed protocol: multiple streams per connection, header compression |
| 6 | Stream in HTTP/2? | Virtual channel within TCP connection; each RPC = one stream |
| 7 | Frame in HTTP/2? | Atomic unit of data: HEADERS, DATA, SETTINGS, RST_STREAM |
| 8 | Why HTTP/2 multiplexing matters? | Many concurrent RPCs on one connection — no head-of-line blocking |
| 9 | What is HPACK? | HTTP/2 header compression — static/dynamic table; repeating headers cost almost nothing |
| 10 | Head-of-line blocking? | HTTP/1.1: large response blocks subsequent requests on same connection; HTTP/2 solves this |

### Protocol Buffers

| # | Question | Answer |
|---|----------|--------|
| 11 | `syntax = "proto3"` meaning? | Proto3 syntax: all fields optional, no required keyword, default values |
| 12 | Proto3 default values? | int=0, float=0.0, bool=false, string="", bytes="", enum=0, message=null |
| 13 | Why enum must start at 0? | Proto3 requires 0 as default; define `UNSPECIFIED=0` |
| 14 | What is a `message`? | Struct/class with typed fields and field numbers |
| 15 | What is a `service`? | Interface with RPC methods and their request/response types |
| 16 | What is `repeated`? | List field: `repeated string tags = 1` → `list[str]` |
| 17 | What is `oneof`? | Union — only one field in the group can be set |
| 18 | What is `optional` in proto3? | Explicit presence — distinguishes unset from default value |
| 19 | What is `map`? | Key-value: `map<string, double> hyperparams = 1` |
| 20 | What is `reserved`? | Block field numbers/names permanently from reuse |
| 21 | What is `import`? | Include another proto file: `import "google/protobuf/timestamp.proto"` |
| 22 | What is a `package`? | Namespace for messages: `package mlplatform.v1;` |
| 23 | Can you rename a field? | Yes — wire format uses field number, renaming is safe |
| 24 | Can you change a field number? | Never — breaks all existing clients |
| 25 | Can you add a new field? | Yes — old clients ignore unknown fields (forward compat) |
| 26 | Can you remove a field? | Yes — mark as `reserved` to prevent reuse |
| 27 | What are well-known types? | Google's pre-built: `Timestamp`, `Duration`, `Empty`, `Any`, `Struct` |
| 28 | `google.protobuf.Empty`? | Empty message — return type for void RPCs |
| 29 | How to version protobuf APIs? | Package versioning: `mlplatform.v1`, `mlplatform.v2` |
| 30 | Proto2 vs proto3? | Proto2: explicit required/optional; proto3: all optional, simpler, preferred |

### Communication Types

| # | Question | Answer |
|---|----------|--------|
| 31 | 4 gRPC communication types? | Unary, Server Streaming, Client Streaming, Bidirectional Streaming |
| 32 | Unary RPC? | One request → one response. Standard synchronous call |
| 33 | Server streaming use cases? | LLM token gen, log tailing, training progress, real-time events |
| 34 | Client streaming use cases? | Large file upload, batch sensor data, audio transcription |
| 35 | Bidir streaming use cases? | Real-time chat, collaborative editing, interactive voice agents |
| 36 | Server streaming proto? | `rpc Watch (WatchReq) returns (stream Event);` |
| 37 | Client streaming proto? | `rpc Upload (stream DataChunk) returns (UploadResp);` |
| 38 | Bidir streaming proto? | `rpc Chat (stream Message) returns (stream Message);` |
| 39 | How to end server stream? | Server's async generator returns / raises StopAsyncIteration |
| 40 | Client cancel stream? | `context.cancel()` — server sees `context.cancelled()` = True |

### Status Codes and Errors

| # | Question | Answer |
|---|----------|--------|
| 41 | Key gRPC status codes? | NOT_FOUND, INVALID_ARGUMENT, PERMISSION_DENIED, UNAUTHENTICATED, UNAVAILABLE, DEADLINE_EXCEEDED |
| 42 | DEADLINE_EXCEEDED? | RPC exceeded deadline; retry with backoff |
| 43 | UNAVAILABLE? | Service temporarily down; safe to retry |
| 44 | NOT_FOUND vs INVALID_ARGUMENT? | NOT_FOUND=resource absent; INVALID_ARGUMENT=bad input |
| 45 | UNAUTHENTICATED vs PERMISSION_DENIED? | UNAUTHENTICATED=no creds; PERMISSION_DENIED=valid creds, not authorized |
| 46 | Abort in Python? | `context.abort(grpc.StatusCode.NOT_FOUND, "message")` |
| 47 | Error in Go? | `status.Errorf(codes.NotFound, "user %d not found", id)` |
| 48 | Rich error details? | `google.rpc.Status` with `details` — include `BadRequest`, `RetryInfo` |
| 49 | Retryable vs non-retryable? | Retry: UNAVAILABLE, DEADLINE_EXCEEDED; Don't retry: INVALID_ARGUMENT, NOT_FOUND |
| 50 | How to pass error metadata? | `context.set_trailing_metadata([("x-error-code", "QUOTA_EXCEEDED")])` |

### Performance

| # | Question | Answer |
|---|----------|--------|
| 51 | Why protobuf faster than JSON? | Binary encoding, varint integers, field numbers not names, no nulls |
| 52 | Varint encoding? | Small integers use fewer bytes: 0-127 in 1 byte |
| 53 | Payload size comparison? | Protobuf ≈ 3-10× smaller than equivalent JSON |
| 54 | gRPC compression? | `grpc.Compression.GZIP` per-call or per-channel |
| 55 | When gRPC is NOT faster? | Very small payloads where HTTP/2 handshake dominates |
| 56 | Multiplexing benefit? | Many concurrent RPCs on one TCP — no per-RPC connection cost |
| 57 | Flow control? | HTTP/2 window — sender pauses when window exhausted; prevents OOM |
| 58 | Connection reuse? | One TCP+TLS handshake for all RPCs — significant latency savings |
| 59 | Keepalive? | Periodically ping to detect dead connections; set `keepalive_time_ms` |
| 60 | Max message size? | Default 4MB; set `max_send_message_length` and `max_receive_message_length` |

### Security

| # | Question | Answer |
|---|----------|--------|
| 61 | TLS in gRPC? | `ssl_channel_credentials(root_certificates=ca_cert)` |
| 62 | mTLS in gRPC? | Both `private_key` and `certificate_chain` in credentials |
| 63 | Auth interceptor? | `grpc.aio.ServerInterceptor` — validate token from metadata |
| 64 | Pass auth in gRPC? | `metadata=[("authorization", "Bearer token")]` in stub call |
| 65 | Token refresh? | `grpc.AuthMetadataPlugin` — called before each RPC for fresh token |
| 66 | Certificate rotation? | cert-manager auto-rotates; gRPC supports reloading credentials |
| 67 | Input validation? | Type safety from proto; `protoc-gen-validate` for field constraints |
| 68 | gRPC reflection security? | Disable in production — exposes all proto definitions |
| 69 | Audit logging? | Interceptor logs method, user, request hash |
| 70 | SPIFFE? | Secure Production Identity Framework — automatic cert provisioning |

### Microservices and Operations

| # | Question | Answer |
|---|----------|--------|
| 71 | gRPC health protocol? | `grpc.health.v1.Health.Check` returns SERVING/NOT_SERVING |
| 72 | K8s liveness probe? | `grpc_health_probe` or native K8s gRPC probe (v1.24+) |
| 73 | LB in K8s? | L4 (kube-proxy) won't balance streams — need L7 (Envoy/Istio) |
| 74 | Client-side LB? | Client maintains server list, distributes RPCs — no proxy needed |
| 75 | Service config? | JSON retry/LB/timeout policy per method; pushed via service discovery |
| 76 | Retry policy? | Max attempts, retryable status codes, backoff — declared in service config |
| 77 | What is a deadline? | Absolute time RPC must complete; propagated across service hops |
| 78 | Why always set deadline? | Without it, stuck RPC holds resource forever — cascade failures |
| 79 | Deadline propagation? | Extract from incoming context; pass to all downstream calls |
| 80 | grpc-gateway? | `protoc-gen-grpc-gateway` — generates HTTP/JSON proxy from proto annotations |

### Streaming Patterns

| # | Question | Answer |
|---|----------|--------|
| 81 | LLM streaming with gRPC? | Server streaming: yield tokens as generated |
| 82 | Backpressure in gRPC streams? | HTTP/2 window fills — server blocks on yield until consumer reads |
| 83 | Detect stream disconnection? | Check `context.cancelled()` before each yield |
| 84 | Bidir message ordering? | No cross-stream ordering guarantee — use sequence numbers if needed |
| 85 | Large file upload? | Client streaming: split into chunks, stream `FileChunk` messages |
| 86 | Channel vs aio.Channel? | Sync vs async client — use `aio` in async Python services |
| 87 | Channel lifecycle? | Reuse channels — creating per-request is expensive |
| 88 | gRPC stream vs WebSocket? | gRPC: typed, codegen, HTTP/2; WS: browser-native, simpler pub/sub |
| 89 | Context propagation in streams? | Pass trace context in initial metadata |
| 90 | How to test gRPC? | Start server on random port in test; use generated stub to call it |

### Observability and Advanced

| # | Question | Answer |
|---|----------|--------|
| 91 | OpenTelemetry for gRPC? | `opentelemetry-instrumentation-grpc` auto-traces all methods |
| 92 | Metadata to log? | Method, user ID, request ID, status code, latency |
| 93 | Trace across services? | Propagate `traceparent` in metadata |
| 94 | What is grpcurl? | CLI tool to call gRPC servers (like curl for REST) |
| 95 | What is Buf? | Modern proto toolchain: linting, breaking change detection, BSR |
| 96 | Buf breaking detection? | `buf breaking --against .git#branch=main` — CI safety check |
| 97 | Proto linting? | `buf lint` — naming conventions, enum UNSPECIFIED, service naming |
| 98 | What is BSR? | Buf Schema Registry — central proto module registry |
| 99 | gRPC in browsers? | grpc-web — HTTP/1.1 or HTTP/2 with Envoy proxy |
| 100 | REST→gRPC migration? | Deploy grpc-gateway; serve both; gradually migrate clients |

## Top System Design Questions (with Answers)

| # | Question | Key Points |
|---|----------|-----------|
| 1 | Design a URL shortener | Hash URL→short code; Redis for hot lookups; 301 (permanent) vs 302 (temporary, trackable); consistent hashing for distributed key space |
| 2 | Design a rate limiter | Token bucket per API key in Redis; sliding window for accuracy; return 429 + Retry-After; distributed with Lua scripts for atomicity |
| 3 | Design an LLM serving platform | vLLM/TGI for batching; FastAPI streaming (SSE); GPU auto-scaling; model versioning; A/B routing at gateway |
| 4 | Design a notification system | Kafka for fan-out; idempotent delivery; per-channel consumers (email, SMS, push); at-least-once + deduplication at receiver |
| 5 | Design an API gateway | Auth (JWT verify), rate limit (Redis), routing (consistent hash by tenant), load balancing, circuit breaker, observability |

## Top AI Engineering Questions (with Answers)

| # | Question | Key Points |
|---|----------|-----------|
| 1 | Design a RAG system API | Embed→vector search→rerank→LLM; streaming response; cache popular queries; metadata filtering |
| 2 | Model serving latency optimization | Batching (dynamic batch in vLLM), quantization (AWQ/GPTQ), KV cache, continuous batching, speculative decoding |
| 3 | Feature store design | Online (Redis, low-latency) + offline (Parquet, batch training); feature versioning; point-in-time correctness |
| 4 | A/B testing models | Route % traffic via feature flag at gateway; log request+response+label; compute metrics; gradual rollout |
| 5 | Agent API design | Stateful sessions (LangGraph checkpointer); streaming events; tool call handling; interrupt+resume for HITL |

---

# 11. Learning Roadmap

## Beginner (0–3 months)

**Goal:** Understand HTTP fundamentals, build simple REST APIs

| Topic | Resource / Exercise |
|-------|-------------------|
| HTTP methods, status codes, headers | Read MDN HTTP docs; use curl to inspect headers |
| Build a simple REST API | FastAPI CRUD: users + posts with SQLite |
| JWT authentication | Add JWT login/logout to the CRUD app |
| OpenAPI/Swagger | Explore auto-generated docs from FastAPI |
| Basic testing | Write pytest tests with `AsyncClient` |

**Projects:**
- Blog API: CRUD for posts, comments, users with JWT auth
- Todo API with pagination, filtering, and sorting

## Intermediate (3–6 months)

**Goal:** Production-ready REST + FastAPI + intro to gRPC

| Topic | Exercise |
|-------|---------|
| Async SQLAlchemy + PostgreSQL | Replace SQLite with async Postgres, connection pooling |
| Redis caching + rate limiting | Cache `GET /users/{id}`; rate limit by API key |
| Middleware, background tasks | Logging middleware, async email notifications |
| gRPC basics | Write first `.proto`, implement unary RPC in Python and Go |
| Docker + Kubernetes | Containerize FastAPI app; deploy with Helm |
| Observability | Add OpenTelemetry tracing + Prometheus metrics |

**Projects:**
- Multi-service app: User service (FastAPI/REST) + Notification service (gRPC)
- Rate-limited API with Redis, JWT, and OpenTelemetry

## Advanced (6–12 months)

**Goal:** Distributed systems patterns, high-performance APIs, ML serving

| Topic | Exercise |
|-------|---------|
| gRPC streaming | LLM token streaming via server streaming RPC |
| Circuit breakers + retries | Implement `CircuitBreaker` class in production service |
| Distributed transactions / Saga | Order service + Payment service with compensating transactions |
| API gateway design | Build a simple gateway with auth, rate limit, routing |
| ML model serving | FastAPI + vLLM streaming completions endpoint |
| Feature store API | Build online feature serving with Redis + FastAPI |

**Projects:**
- LLM serving platform: FastAPI + vLLM + streaming + rate limiting + observability
- gRPC feature store with Redis backend and Python + Go clients

## Expert (12–18 months)

**Goal:** Principal-level system design, cross-team architecture decisions

| Topic | Focus |
|-------|-------|
| API platform design | Multi-tenant, versioned, globally distributed API platform |
| gRPC service mesh | Envoy/Istio with mTLS, traffic management, circuit breaking |
| Large-scale ML serving | Multi-model routing, shadow deployment, canary rollout |
| Protocol design | Custom protocol on top of HTTP/2; when to build vs buy |
| Failure mode analysis | Chaos engineering; failure injection; runbooks |

**Mock Interview Topics:**
- "Design the OpenAI API" — rate limiting, streaming, billing, versioning
- "Design a real-time ML feature store" — latency, consistency, scale
- "How would you migrate 50 REST services to gRPC?" — phasing, grpc-gateway, tooling
- "Design a multi-tenant FastAPI platform" — isolation, routing, observability

---

## Quality Checklist Verification

| Checklist Item | Status |
|----------------|--------|
| No important REST topic missed | Covered: history, constraints, methods, status codes, headers, auth, versioning, design, challenges |
| No important FastAPI topic missed | Covered: routing, DI, async, security, DB, middleware, testing, observability, deployment |
| No important gRPC topic missed | Covered: proto, 4 comm types, HTTP/2, security, microservices, observability |
| Python examples included | Yes — all major patterns have Python code |
| Go examples included | Yes — HTTP client, gRPC server, gRPC client |
| Mermaid diagrams included | Yes — 15+ diagrams |
| Production considerations | Yes — connection pooling, rate limiting, observability, backpressure |
| Security covered | JWT, OAuth2, mTLS, RBAC, ABAC, rate limiting, CORS |
| Testing covered | pytest, AsyncClient, dependency overrides, mocking |
| Deployment covered | Docker, Kubernetes, Uvicorn workers, CI/CD |
| Observability covered | OpenTelemetry, Prometheus, Jaeger, structured logging, correlation IDs |
| System Design covered | API Gateway, LB, caching, message queues, service mesh |
| AI Engineering covered | LLM serving, RAG, feature store, MCP servers, model gateway |
| Interview Questions | 20+ per section + 50 in crash revision |
| Failure Scenarios | 14 failure types with diagnosis and remediation |
| Learning Roadmap | 4 levels from Beginner to Expert |

---

*End of REST APIs · FastAPI · gRPC Interview Guide — mid-2026 edition.*
