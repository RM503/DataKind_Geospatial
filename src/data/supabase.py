from __future__ import annotations

import os

from supabase import Client, create_client

_supabase_client: Client | None = None

def get_supabase_client() -> Client:
    global _supabase_client

    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        _supabase_client = create_client(url, key)

    return _supabase_client