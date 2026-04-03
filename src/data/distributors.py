from __future__ import annotations

import pandas as pd

from .supabase import get_supabase_client
from common.logging_config import get_logger

logger = get_logger(__name__)

def get_distributor_data() -> pd.DataFrame:
    client = get_supabase_client()

    try:
        response = (
            client.table("distributor_locations")
                  .select("*")
                  .execute()
        )

        if response.data is None:
            raise ValueError("Supabase returned no data.")

        return pd.DataFrame(response.data)

    except Exception as e:
        logger.exception("Failed to fetch distributor locations from Supabase.")
        raise e