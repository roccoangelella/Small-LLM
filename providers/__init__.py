"""Shared execution adapters used by provider-specific launchers.

Provider SDK imports deliberately stay in ``beam/`` and ``modal/``.  This
package owns only code whose behavior must remain identical across providers.
"""
