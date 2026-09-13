import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main


def paper(title: str, abstract: str = "") -> main.Paper:
    return main.Paper("2609.12345", title, ["A. Researcher"], abstract,
        "https://arxiv.org/abs/2609.12345", "https://arxiv.org/pdf/2609.12345",
        "2026-09-12T00:00:00+00:00", "2026-09-12T00:00:00+00:00", ["cs.CV"])


class Tests(unittest.TestCase):
    def test_acronym_boundary(self):
        self.assertTrue(main.phrase_present("VLM", "A compact VLM"))
        self.assertFalse(main.phrase_present("VLM", "unrelatedvlmtext"))

    def test_title_weight(self):
        score, topics = main.score_paper(paper("Multimodal Agents"),
            {"多模态": {"include": ["multimodal"], "exclude": []}})
        self.assertEqual((score, topics), (3, ["多模态"]))

    def test_fenced_json(self):
        self.assertEqual(main.parse_json_object('```json\n{"papers": []}\n```'), {"papers": []})

    def test_seen_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seen.json"
            main.save_seen(path, {"2609.00001"})
            self.assertEqual(main.load_seen(path), {"2609.00001"})

    @patch("main.smtplib.SMTP_SSL")
    def test_email_defaults(self, smtp_ssl):
        smtp = smtp_ssl.return_value.__enter__.return_value
        with patch.dict(os.environ, {"EMAIL_SENDER": "", "EMAIL_RECIPIENT": "", "EMAIL_APP_PASSWORD": "x"}, clear=True):
            main.send_email("subject", "body", "digest.md")
        smtp.login.assert_called_once_with("yangxue7410@gmail.com", "x")

    def test_sensenova_chat_completions(self):
        client = Mock(); response = client.chat.completions.create.return_value
        response.choices = [Mock()]
        response.choices[0].message.content = json.dumps({"papers": []})
        self.assertEqual(main.analyze_chunk(client, "sensenova-6.8-flash-lite", [paper("A VLM")]), [])
        args = client.chat.completions.create.call_args.kwargs
        self.assertEqual(args["model"], "sensenova-6.8-flash-lite")

    def test_cli_config_maps_to_run_parameter(self):
        with patch("sys.argv", ["main.py", "--config", "custom.yaml", "--dry-run"]):
            args = main.parse_args()
        self.assertEqual(args.config_path, Path("custom.yaml"))
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
