import sys
import os
from pathlib import Path

current_dir = Path(__file__).parent
src_dir = current_dir / "src" / "python"

sys.path.insert(0, str(src_dir))
os.chdir(src_dir)

if __name__ == "__main__":
    try:
        from core.hft_integration import main
        import asyncio
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\n⚠️  System interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()