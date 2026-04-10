"""Dataset loading utilities for benchmark corpora and questions."""
import json
import logging
import os
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class DatasetLoader:
    @staticmethod
    def prepare_medical(datasets_root: str, output_dir: str) -> Optional[List[Dict[str, str]]]:
        """Load medical corpus and questions from GraphRAG-Benchmark/Datasets/.

        Writes corpus text as a single .txt file into output_dir.
        Returns questions list (with ground_truth) or None on error.
        """
        try:
            corpus_path = os.path.join(datasets_root, "Corpus", "medical.json")
            if not os.path.isfile(corpus_path):
                log.warning(f"Medical corpus not found: {corpus_path}")
                return None
            os.makedirs(output_dir, exist_ok=True)
            with open(corpus_path, encoding="utf-8") as f:
                corpus = json.load(f)
            context = corpus.get("context", "")
            with open(os.path.join(output_dir, "medical.txt"), "w", encoding="utf-8") as fout:
                fout.write(context)

            questions_path = os.path.join(datasets_root, "Questions", "medical_questions.json")
            if not os.path.isfile(questions_path):
                log.warning(f"Medical questions not found: {questions_path}")
                return None
            with open(questions_path, encoding="utf-8") as f:
                raw_questions = json.load(f)
            questions = []
            for q in raw_questions:
                questions.append({
                    "question_id": q.get("id", ""),
                    "question_text": q.get("question", ""),
                    "ground_truth": q.get("answer", ""),
                    "evidence": q.get("evidence", ""),
                    "question_type": q.get("question_type", ""),
                    "source": q.get("source", "Medical"),
                })
            log.info(f"Medical: {len(questions)} questions loaded with ground_truth")
            return questions
        except Exception as e:
            log.warning(f"Failed to prepare medical: {e}")
            return None

    @staticmethod
    def prepare_novel(datasets_root: str, output_dir: str) -> Optional[List[Dict[str, str]]]:
        """Load novel corpus and questions from GraphRAG-Benchmark/Datasets/.

        Writes each novel as a separate .txt file into output_dir.
        Returns questions list (with ground_truth) or None on error.
        """
        try:
            corpus_path = os.path.join(datasets_root, "Corpus", "novel.json")
            if not os.path.isfile(corpus_path):
                log.warning(f"Novel corpus not found: {corpus_path}")
                return None
            os.makedirs(output_dir, exist_ok=True)
            with open(corpus_path, encoding="utf-8") as f:
                corpus_list = json.load(f)
            for item in corpus_list:
                name = item.get("corpus_name", "unknown")
                context = item.get("context", "")
                with open(os.path.join(output_dir, f"{name}.txt"), "w", encoding="utf-8") as fout:
                    fout.write(context)

            questions_path = os.path.join(datasets_root, "Questions", "novel_questions.json")
            if not os.path.isfile(questions_path):
                log.warning(f"Novel questions not found: {questions_path}")
                return None
            with open(questions_path, encoding="utf-8") as f:
                raw_questions = json.load(f)
            questions = []
            for q in raw_questions:
                questions.append({
                    "question_id": q.get("id", ""),
                    "question_text": q.get("question", ""),
                    "ground_truth": q.get("answer", ""),
                    "evidence": q.get("evidence", ""),
                    "question_type": q.get("question_type", ""),
                    "source": q.get("source", ""),
                })
            log.info(f"Novel: {len(questions)} questions loaded with ground_truth")
            return questions
        except Exception as e:
            log.warning(f"Failed to prepare novel: {e}")
            return None
