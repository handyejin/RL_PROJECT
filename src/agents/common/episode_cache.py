"""구/날짜별 episode 로딩 cache.

원본 parquet에서 episode를 매번 다시 만들면 구별 학습 시작 전 로딩 시간이 길다.
이 helper는 `load_episode()` 결과를 pickle로 저장해 같은 processed data, 구, 날짜
조건에서는 다음 실행부터 바로 재사용한다.

주의:
    capacity override, forecast override, DQN reward scale은 cache에 넣지 않는다.
    cache에는 순수 episode만 저장하고, 각 agent core가 기존처럼 로딩 후 override를
    적용한다. 따라서 forecast 파일이나 capacity 파일을 바꿔도 episode cache를
    재사용할 수 있다.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm


def _cache_key(processed_dir: str, district: str, date: str) -> str:
    """episode cache 파일명에 사용할 짧은 hash key를 만든다."""
    raw = f"{Path(processed_dir).resolve()}|{district}|{date}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def load_episodes_cached(
    dates: list[str],
    district: str,
    processed_dir: str,
    loader: Callable[[str, str, str], object],
    cache_dir: str | None = "data/episode_cache",
    progress_label: str | None = None,
) -> list:
    """날짜 목록을 episode로 변환하되, 있으면 cache에서 먼저 읽는다.

    Args:
        dates: `YYYY-MM-DD` 날짜 목록.
        district: 실행 구 이름.
        processed_dir: parquet 전처리 데이터 경로.
        loader: cache miss 때 호출할 함수. `(processed_dir, district, date)`를 받는다.
        cache_dir: cache 저장 경로. 빈 문자열이나 None이면 cache를 사용하지 않는다.
        progress_label: tqdm에 표시할 문구.

    Returns:
        날짜 순서와 같은 episode list.
    """
    if not cache_dir:
        if progress_label:
            with tqdm(dates, desc=progress_label, unit="day") as iterator:
                return [loader(processed_dir, district, date) for date in iterator]
        return [loader(processed_dir, district, date) for date in dates]

    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)

    episodes = []
    hits = 0
    misses = 0
    iterator_context = tqdm(dates, desc=progress_label, unit="day") if progress_label else None
    iterator = iterator_context if iterator_context is not None else dates
    for date in iterator:
        key = _cache_key(processed_dir, district, date)
        path = root / f"{district}_{date}_{key}.pkl"
        if path.exists():
            with path.open("rb") as f:
                episode = pickle.load(f)
            hits += 1
        else:
            episode = loader(processed_dir, district, date)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(episode, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
            misses += 1
        episodes.append(episode)
    if iterator_context is not None:
        iterator_context.close()

    if progress_label:
        print(f"\n{progress_label} cache: hit={hits}, miss={misses}, dir={root}")
    return episodes
