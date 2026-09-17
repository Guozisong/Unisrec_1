import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_pipeline.raw.get_data_from_odps import connect_odps


class ODPSConnectionTest(unittest.TestCase):
    def test_reads_all_connection_fields_from_selected_env_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text(
                "access_id=test-id\n"
                "access_key=test-key\n"
                "project=test-project\n"
                "endpoint=https://example.invalid/api\n"
            )
            with patch("data_pipeline.raw.get_data_from_odps.ODPS") as odps:
                connect_odps(env_file)
            odps.assert_called_once_with(
                "test-id", "test-key", "test-project", endpoint="https://example.invalid/api"
            )

    def test_rejects_missing_fields_without_creating_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_file = Path(temporary) / ".env"
            env_file.write_text("access_id=test-id\n")
            with patch("data_pipeline.raw.get_data_from_odps.ODPS") as odps:
                with self.assertRaisesRegex(ValueError, "access_key, project, endpoint"):
                    connect_odps(env_file)
            odps.assert_not_called()


if __name__ == "__main__":
    unittest.main()
