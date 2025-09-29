"""GraphQL schema for event data service."""

import strawberry
from .resolvers import Query, Mutation


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
)