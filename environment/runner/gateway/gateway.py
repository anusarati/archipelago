"""Core MCP gateway logic for building and hot-swapping MCP apps.

This module handles creating FastMCP proxy ASGI apps and hot-swapping them
in the FastAPI application without restarting the server.
"""

import asyncio
import contextlib
import inspect
import os
import time

from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from fastmcp import Client as FastMCPClient
from fastmcp import FastMCP
from fastmcp.server.http import StarletteWithLifespan
from loguru import logger
from starlette.routing import Mount

from .models import (
    MCPSchema,
    ServerReadinessDetails,
)
from .state import (
    get_mcp_lifespan_manager,
    get_mcp_lock,
    get_mcp_mount,
    set_mcp_lifespan_manager,
    set_mcp_mount,
)


class MCPReadinessError(Exception):
    """Exception raised when MCP servers fail readiness check.

    Attributes:
        failed_servers: Dict mapping server names to readiness details
        message: Human-readable error message
    """

    failed_servers: dict[str, ServerReadinessDetails]
    message: str

    def __init__(
        self,
        failed_servers: dict[str, ServerReadinessDetails],
        message: str | None = None,
    ):
        """Initialize MCP readiness error.

        Args:
            failed_servers: Dict mapping server names to ServerReadinessDetails
            message: Optional custom error message
        """
        self.failed_servers = failed_servers
        server_list = ", ".join(failed_servers.keys())
        self.message = message or f"MCP servers not ready: {server_list}"
        super().__init__(self.message)


def _parse_float_env(name: str, default: float) -> float:
    """Parse float from environment variable with a safe fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            f"Invalid value for {name}={raw!r}; using default {default:.1f}"
        )
        return default


async def _resolve_mounted_servers_for_probe(mcp_proxy: FastMCP) -> list[object]:
    """Resolve mounted servers used by FastMCP's aggregate list_tools path.

    For FastMCP proxies created from MCP config dicts, mounted servers live
    under MCPConfigTransport._composite_server rather than mcp_proxy itself.
    """
    direct_mounted = getattr(mcp_proxy, "_mounted_servers", None)
    if isinstance(direct_mounted, list) and direct_mounted:
        return direct_mounted

    client_factory = getattr(mcp_proxy, "client_factory", None)
    if client_factory is None:
        return []

    try:
        maybe_client = client_factory()
        if inspect.isawaitable(maybe_client):
            maybe_client = await maybe_client
        transport = getattr(maybe_client, "transport", None)
        composite_server = getattr(transport, "_composite_server", None)
        composite_mounted = getattr(composite_server, "_mounted_servers", None)
        if isinstance(composite_mounted, list):
            return composite_mounted
    except Exception as e:
        logger.debug(f"Failed to resolve composite mounted servers: {e}")

    return []


def _build_mcp_app_with_proxy(
    config: MCPSchema,
) -> tuple[StarletteWithLifespan, FastMCP | None]:
    """Build a FastMCP proxy ASGI app from MCP configuration.

    Internal function that returns both the HTTP app and the proxy instance.

    Args:
        config: MCP configuration schema with "mcpServers" key

    Returns:
        Tuple of (ASGI app, FastMCP proxy or None if no servers)
    """
    if not config.mcpServers:
        mcp_server = FastMCP(name="Gateway")
        mcp_app = mcp_server.http_app(path="/")
        return mcp_app, None

    # FastMCP's config parser is sensitive to keys being present with null values
    # (e.g., http servers should not also include {"command": null, ...}).
    # Only emit explicitly-set fields.
    config_dict = config.model_dump(exclude_none=True)
    mcp_proxy = FastMCP.as_proxy(config_dict, name="Gateway")

    # Create HTTP ASGI app, root at "/" so final URLs are under /mcp
    mcp_app = mcp_proxy.http_app(path="/")

    return mcp_app, mcp_proxy


async def warm_and_check_gateway(
    mcp_proxy: FastMCP,
    expected_servers: list[str],
    max_wait_seconds: float = 10.0,
    retry_interval: float = 1.0,
) -> int:
    """Warm up gateway and verify all expected servers provide tools.

    Uses parallel per-mounted-server probes to avoid FastMCP's default
    sequential mounted-server listing path during cold start. Falls back to
    aggregate list_tools probing if mounted-server introspection is unavailable.

    Args:
        mcp_proxy: The FastMCP proxy instance to warm up
        expected_servers: List of server names that must provide tools
        max_wait_seconds: Maximum time to wait for all servers (default 10s)
        retry_interval: Time between retry attempts (default 1s)

    Returns:
        Total number of tools loaded across ready servers

    Raises:
        MCPReadinessError: If any server doesn't provide tools within timeout
    """
    if not expected_servers:
        return 0

    start_time = time.perf_counter()
    deadline = start_time + max_wait_seconds
    expected_set = set(expected_servers)

    # Try parallel mounted-server probing first.
    mounted_servers = await _resolve_mounted_servers_for_probe(mcp_proxy)
    server_to_mounted: dict[str, object] = {}
    for mounted in mounted_servers:
        prefix = getattr(mounted, "prefix", None)
        if isinstance(prefix, str) and prefix in expected_set:
            server_to_mounted[prefix] = mounted
            continue

        mounted_server_obj = getattr(mounted, "server", None)
        mounted_name = getattr(mounted_server_obj, "name", None)
        if isinstance(mounted_name, str) and mounted_name in expected_set:
            server_to_mounted[mounted_name] = mounted

    # FastMCP internals can change. If we cannot map every expected server,
    # fall back to aggregate list_tools probing.
    if len(server_to_mounted) < len(expected_servers):
        unresolved = sorted(expected_set - set(server_to_mounted.keys()))
        logger.warning(
            f"Mounted-server mapping incomplete ({len(server_to_mounted)}/{len(expected_servers)}). "
            f"Falling back to aggregate readiness probing. unresolved={unresolved}"
        )
        return await _warm_and_check_gateway_via_aggregate_list_tools(
            mcp_proxy=mcp_proxy,
            expected_servers=expected_servers,
            max_wait_seconds=max_wait_seconds,
            retry_interval=retry_interval,
        )

    async def probe_server(server_name: str, mounted: object) -> tuple[bool, int, int, str]:
        attempts = 0
        last_error = ""
        pending_task: asyncio.Task[list] | None = None

        try:
            while True:
                attempts += 1
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    if not last_error:
                        last_error = "Timeout"
                    return False, 0, attempts, last_error

                mounted_server_obj = getattr(mounted, "server", None)
                list_tools_middleware = getattr(
                    mounted_server_obj, "_list_tools_middleware", None
                )
                if list_tools_middleware is None:
                    return False, 0, attempts, "Mounted server missing _list_tools_middleware"

                if pending_task is None:
                    pending_task = asyncio.create_task(list_tools_middleware())

                poll_timeout = min(retry_interval, remaining)

                try:
                    tools = await asyncio.wait_for(
                        asyncio.shield(pending_task), timeout=poll_timeout
                    )
                    pending_task = None
                    tool_count = len(tools)
                    if tool_count > 0:
                        return True, tool_count, attempts, ""
                    last_error = "No tools returned"
                    logger.debug(
                        f"Server '{server_name}' attempt {attempts}: no tools returned yet"
                    )
                except TimeoutError:
                    last_error = "Timeout"
                    elapsed = time.perf_counter() - start_time
                    logger.debug(
                        f"Server '{server_name}' attempt {attempts} ({elapsed:.1f}s): waiting for tool discovery"
                    )
                except Exception as e:
                    last_error = str(e)
                    pending_task = None
                    elapsed = time.perf_counter() - start_time
                    logger.debug(
                        f"Server '{server_name}' attempt {attempts} ({elapsed:.1f}s): probe failed: {e}"
                    )
        finally:
            if pending_task is not None and not pending_task.done():
                pending_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pending_task

    probe_tasks = {
        server_name: asyncio.create_task(probe_server(server_name, mounted))
        for server_name, mounted in server_to_mounted.items()
    }

    probe_results = await asyncio.gather(*probe_tasks.values())

    elapsed = time.perf_counter() - start_time
    servers_with_tools: dict[str, int] = {}
    failed_servers: dict[str, ServerReadinessDetails] = {}

    for server_name, (is_ready, tool_count, attempts, last_error) in zip(
        probe_tasks.keys(), probe_results
    ):
        if is_ready:
            servers_with_tools[server_name] = tool_count
            logger.info(
                f"Server '{server_name}' ready after {attempts} attempt(s) ({elapsed:.1f}s): {tool_count} tools"
            )
            continue

        error_msg = f"No tools found after {elapsed:.1f}s"
        if last_error:
            error_msg += f" (last error: {last_error})"
        failed_servers[server_name] = ServerReadinessDetails(
            error=error_msg,
            attempts=attempts,
        )
        logger.warning(
            f"Server '{server_name}' FAILED after {attempts} attempt(s) ({elapsed:.1f}s): {error_msg}"
        )

    if not failed_servers:
        total_tools = sum(servers_with_tools.values())
        logger.info(
            f"Gateway ready after {elapsed:.1f}s: {total_tools} tools from {len(expected_servers)} servers"
        )
        return total_tools

    failed_count = len(failed_servers)
    ready_count = len(servers_with_tools)
    logger.error(
        f"MCP readiness check failed: {failed_count} server(s) not ready ({ready_count} server(s) ready)"
    )
    failed_list = ", ".join(sorted(failed_servers.keys()))
    raise MCPReadinessError(
        failed_servers, message=f"MCP servers not ready after {elapsed:.1f}s: {failed_list}"
    )


async def _warm_and_check_gateway_via_aggregate_list_tools(
    mcp_proxy: FastMCP,
    expected_servers: list[str],
    max_wait_seconds: float,
    retry_interval: float,
) -> int:
    """Fallback readiness path using aggregate gateway list_tools()."""
    start_time = time.perf_counter()
    deadline = start_time + max_wait_seconds
    attempts = 0
    last_error: str = ""
    missing_servers: set[str] = set(expected_servers)
    servers_with_tools: dict[str, int] = {}

    # FastMCP only prefixes tools when there are multiple servers
    single_server = len(expected_servers) == 1

    pending_list_tools: asyncio.Task[list] | None = None
    try:
        async with FastMCPClient(mcp_proxy) as client:
            while True:
                attempts += 1
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    last_error = "Timeout"
                    break

                if pending_list_tools is None:
                    pending_list_tools = asyncio.create_task(client.list_tools())

                poll_timeout = min(retry_interval, remaining)

                try:
                    tools = await asyncio.wait_for(
                        asyncio.shield(pending_list_tools), timeout=poll_timeout
                    )
                    pending_list_tools = None
                    tool_names = [t.name for t in tools]

                    servers_with_tools = {}
                    if single_server:
                        server = expected_servers[0]
                        if tool_names:
                            servers_with_tools[server] = len(tool_names)
                    else:
                        sorted_servers = sorted(expected_servers, key=len, reverse=True)
                        claimed_tools: set[str] = set()
                        for server in sorted_servers:
                            prefix = f"{server}_"
                            matching = [
                                name
                                for name in tool_names
                                if name.startswith(prefix) and name not in claimed_tools
                            ]
                            if matching:
                                servers_with_tools[server] = len(matching)
                                claimed_tools.update(matching)

                    missing_servers = set(expected_servers) - set(
                        servers_with_tools.keys()
                    )
                    if not missing_servers:
                        elapsed = time.perf_counter() - start_time
                        total_tools = len(tools)
                        logger.info(
                            f"Gateway ready after {elapsed:.1f}s via aggregate probe: {total_tools} tools from {len(expected_servers)} servers"
                        )
                        return total_tools
                except TimeoutError:
                    last_error = "Timeout"
                except Exception as e:
                    last_error = str(e)
                    pending_list_tools = None
    finally:
        if pending_list_tools is not None and not pending_list_tools.done():
            pending_list_tools.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending_list_tools

    elapsed = time.perf_counter() - start_time
    failed_servers: dict[str, ServerReadinessDetails] = {}

    for server in missing_servers:
        error_msg = f"No tools found after {elapsed:.1f}s"
        if last_error:
            error_msg += f" (last error: {last_error})"
        failed_servers[server] = ServerReadinessDetails(
            error=error_msg,
            attempts=attempts,
        )

    failed_list = ", ".join(sorted(failed_servers.keys()))
    raise MCPReadinessError(
        failed_servers, message=f"MCP servers not ready after {elapsed:.1f}s: {failed_list}"
    )


async def swap_mcp_app(config: MCPSchema, app: FastAPI) -> None:
    """Hot-swap the mounted MCP app with a new configuration.

    This function:
    1. Builds a new MCP app from config
    2. Starts its lifespan
    3. Atomically replaces the Mount.app reference
    4. Shuts down the old app's lifespan
    5. Warms up gateway connections and verifies all servers are ready

    Args:
        config: New MCP configuration schema (MCPSchema instance)
        app: The FastAPI application instance

    Raises:
        ValueError: If config is invalid
        RuntimeError: If swap fails
        MCPReadinessError: If any server fails readiness check
    """
    async with get_mcp_lock():  # Prevent concurrent swaps
        new_app, mcp_proxy = _build_mcp_app_with_proxy(config)

        new_lm = LifespanManager(new_app)
        _ = await new_lm.__aenter__()

        try:
            current_mount = get_mcp_mount()
            if current_mount is None:
                app.mount("/mcp", new_app)

                mount = next(
                    (
                        r
                        for r in app.router.routes
                        if isinstance(r, Mount) and r.path == "/mcp"
                    ),
                    None,
                )
                if mount is None:
                    msg = (
                        "Failed to find mounted MCP gateway after mounting. "
                        "This should not happen and indicates a bug."
                    )
                    raise RuntimeError(msg)
                set_mcp_mount(mount)
            else:
                current_mount.app = new_app

            old_lm = get_mcp_lifespan_manager()
            if old_lm is not None:
                _ = await old_lm.__aexit__(None, None, None)

            set_mcp_lifespan_manager(new_lm)

            server_count = len(config.mcpServers)
            logger.info(
                f"Successfully swapped MCP gateway with {server_count} server(s)"
            )

            if not config.mcpServers or mcp_proxy is None:
                logger.debug("No MCP servers configured, skipping readiness check")
                return

            logger.debug("Waiting 1.0 seconds before starting readiness checks...")
            await asyncio.sleep(1.0)

            server_names = list(config.mcpServers.keys())
            readiness_timeout = _parse_float_env("MCP_READINESS_TIMEOUT_SECONDS", 10.0)
            readiness_poll_interval = _parse_float_env(
                "MCP_READINESS_RETRY_INTERVAL_SECONDS", 1.0
            )
            _ = await warm_and_check_gateway(
                mcp_proxy,
                server_names,
                max_wait_seconds=readiness_timeout,
                retry_interval=readiness_poll_interval,
            )

        except MCPReadinessError:
            raise
        except Exception as e:
            _ = await new_lm.__aexit__(None, None, None)
            logger.error(f"Failed to swap MCP gateway: {e}")
            raise RuntimeError(f"Failed to swap MCP gateway: {e}") from e
