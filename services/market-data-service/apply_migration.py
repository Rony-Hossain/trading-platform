import asyncio
import asyncpg
import sys

async def apply_migrations():
    try:
        # Connect using root .env database
        conn = await asyncpg.connect(
            'postgresql://trading_user:trading_pass@localhost:5432/trading_db'
        )

        print('Connected to database')

        # Read migration file
        with open('db/migrations/20251008_timescale_market_data.sql', 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Execute the migration
        await conn.execute(sql_content)

        await conn.close()

        print('Migrations applied successfully!')
        return True
    except Exception as e:
        print(f'Error applying migrations: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = asyncio.run(apply_migrations())
    sys.exit(0 if success else 1)
