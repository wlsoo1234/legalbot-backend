import os
import asyncio
from google.cloud.alloydb.connector import AsyncConnector
import asyncpg

INSTANCE_URI = "projects/gen-lang-client-0411497779/locations/asia-southeast1/clusters/kitahack-cluster/instances/primary"

DB = "postgres"   # change if needed
# Change this line in test_alloydb.py
USER = "kitahack-learning@gen-lang-client-0411497779.iam.gserviceaccount.com"

async def main():
    print("USER =", USER)
    print("len(USER) =", len(USER))
    connector = AsyncConnector()

    conn = await connector.connect(
        INSTANCE_URI,
        "asyncpg",
        user=USER,
        db=DB,
        enable_iam_auth=False,
        ip_type="PUBLIC",
    )

    version = await conn.fetchval("SELECT version();")
    print("Connected successfully!")
    print(version)
    print("USER =", USER)
    print("len(USER) =", len(USER))

    await conn.close()
    await connector.close()


asyncio.run(main())
