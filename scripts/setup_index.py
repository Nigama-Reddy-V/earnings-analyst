import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

client.create_payload_index(
    collection_name="earnings_transcripts",
    field_name="ticker",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name="earnings_transcripts",
    field_name="session_id",
    field_schema=PayloadSchemaType.KEYWORD
)

print("Indexes created on 'ticker' and 'session_id' fields.")