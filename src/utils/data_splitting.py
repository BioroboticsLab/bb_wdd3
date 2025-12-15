import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.model_selection import KFold
from collections import defaultdict


def get_base_video_name(video_name: str) -> str:
    """
    Extract base video name by removing resolution suffix.
    
    Examples:
        "011_576_440" -> "011"
        "T1-D1-B6-V5-C_480_270" -> "T1-D1-B6-V5-C"
        "074_2304_1760_30fps" -> "074"
    
    Args:
        video_name: Video filename
        
    Returns:
        Base video name without resolution suffix
    """
    # Remove file extension
    name = video_name.replace('.mp4', '').replace('.avi', '')
    
    parts = name.split('_')
    
    if len(parts) >= 3:
        # Check if last part is fps info
        last_idx = -1
        if parts[-1].endswith('fps') and parts[-1][:-3].isdigit():
            last_idx = -2
        
        # Check if two parts before fps are numeric (width and height)
        if len(parts) >= abs(last_idx) + 2:
            width_idx = last_idx - 1
            height_idx = last_idx
            
            if parts[width_idx].isdigit() and parts[height_idx].isdigit():
                return '_'.join(parts[:width_idx])
    
    return name


def get_video_category(base_video_name: str) -> str:
    """
    Categorize video based on naming convention.
    
    Categories:
        - "0": Videos starting with 0
        - "T1": Videos starting with T1
        - "T_other": Videos starting with T(other number)
        - "C": Videos starting with C
        - "other": Everything else
    
    Args:
        base_video_name: Base video name
        
    Returns:
        Category string
    """
    if base_video_name.startswith('0'):
        return "0"
    elif base_video_name.startswith('T'):
        return "T"
    elif base_video_name.startswith('C'):
        return "C"
    else:
        return "other"


def get_fold_statistics(data: pd.DataFrame, video_col: str = 'video_name') -> pd.DataFrame:
    """
    Get statistics for each base video (number of waggle runs, clips, etc.).
    Uses a globally-unique waggle id (waggle_uid = base_video + '_' + waggle_run_id)
    so that waggle_run_id restarting from 1 per video is handled correctly.
    """
    data = data.copy()
    if 'base_video' not in data.columns:
        data['base_video'] = data[video_col].apply(get_base_video_name)
    if 'category' not in data.columns:
        data['category'] = data['base_video'].apply(get_video_category)

    # Create a globally unique waggle id per video
    if 'waggle_run_id' in data.columns:
        # make sure waggle_run_id is string-like
        data['waggle_run_id'] = data['waggle_run_id'].astype(str)
        data['waggle_uid'] = data['base_video'].astype(str) + '_' + data['waggle_run_id']
    else:
        data['waggle_uid'] = None

    # Group by base_video and calculate statistics
    stats_dict = {
        'n_clips': data.groupby('base_video').size(),
        'category': data.groupby('base_video')['category'].first()
    }

    if 'waggle_uid' in data.columns and data['waggle_uid'].notnull().any():
        stats_dict['n_waggle_runs'] = data.groupby('base_video')['waggle_uid'].nunique()

    video_stats = pd.DataFrame(stats_dict).reset_index()
    return video_stats



def split_by_video_groups(
    data: pd.DataFrame,
    train_ratio: float = 0.75,
    seed: int = 42,
    video_col: str = 'video_name',
    min_category_train_videos: int = 1,
    subset_size: float = None,
    subset_videos: int = None,
    subset_per_category: Dict[str, int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by waggle runs (not video count) ensuring:
    1. No data leakage (same base video not in both sets)
    2. Train/val waggle run ratio approximately matches train_ratio (e.g., 75:25)
    3. All categories represented in training set
    
    Args:
        data: DataFrame with annotations (must have 'waggle_run_id' column)
        train_ratio: Target ratio of waggle runs for training (default 0.75)
        seed: Random seed for reproducibility
        video_col: Column name containing video names
        min_category_train_videos: Minimum videos per category in training
        subset_size: Use only this fraction of total videos (e.g., 0.1 for 10%)
        subset_videos: Use only this many total videos (absolute number)
        subset_per_category: Dict mapping category to number of videos to use
        
    Returns:
        Tuple of (train_df, val_df)
    """
    np.random.seed(seed)
    
    # Validate required columns
    if 'waggle_run_id' not in data.columns:
        raise ValueError("Data must contain 'waggle_run_id' column for waggle-based splitting")
    
    # Add base video and category columns
    data = data.copy()
    data['base_video'] = data[video_col].apply(get_base_video_name)
    data['category'] = data['base_video'].apply(get_video_category)


    if 'waggle_run_id' in data.columns:
        data['waggle_run_id'] = data['waggle_run_id'].astype(str)
        data['waggle_uid'] = data['base_video'].astype(str) + '_' + data['waggle_run_id']
    else:
        data['waggle_uid'] = None
    
    # Get video statistics
    video_stats = get_fold_statistics(data, video_col)
    
    # Group videos by category with their waggle counts
    category_video_info = defaultdict(list)
    for _, row in video_stats.iterrows():
        category_video_info[row['category']].append({
            'video': row['base_video'],
            'n_waggles': row.get('n_waggle_runs', 0),
            'n_clips': row['n_clips']
        })
    
    # Apply subset selection if requested
    if subset_per_category is not None:
        print(f"\n{'='*70}")
        print(f"Subset Selection: Per-Category")
        print(f"{'='*70}")
        for cat, n_videos in subset_per_category.items():
            if cat in category_video_info:
                original_count = len(category_video_info[cat])
                videos = category_video_info[cat].copy()
                np.random.shuffle(videos)
                category_video_info[cat] = videos[:min(n_videos, len(videos))]
                print(f"  {cat:10s}: Using {len(category_video_info[cat]):3d}/{original_count:3d} videos")
    
    elif subset_videos is not None:
        print(f"\n{'='*70}")
        print(f"Subset Selection: {subset_videos} Total Videos (Stratified)")
        print(f"{'='*70}")
        
        total_videos = sum(len(videos) for videos in category_video_info.values())
        for cat, videos in category_video_info.items():
            original_count = len(videos)
            proportion = len(videos) / total_videos
            n_videos = max(1, int(subset_videos * proportion))
            
            videos_copy = videos.copy()
            np.random.shuffle(videos_copy)
            category_video_info[cat] = videos_copy[:min(n_videos, len(videos_copy))]
            print(f"  {cat:10s}: Using {len(category_video_info[cat]):3d}/{original_count:3d} videos")
    
    elif subset_size is not None:
        print(f"\n{'='*70}")
        print(f"Subset Selection: {subset_size*100:.1f}% of Videos (Stratified)")
        print(f"{'='*70}")
        
        for cat, videos in category_video_info.items():
            original_count = len(videos)
            n_videos = max(1, int(len(videos) * subset_size))
            
            videos_copy = videos.copy()
            np.random.shuffle(videos_copy)
            category_video_info[cat] = videos_copy[:n_videos]
            print(f"  {cat:10s}: Using {len(category_video_info[cat]):3d}/{original_count:3d} videos")
    
    # Filter data to only include selected videos
    if subset_per_category is not None or subset_videos is not None or subset_size is not None:
        all_selected_videos = set()
        for videos_info in category_video_info.values():
            all_selected_videos.update([v['video'] for v in videos_info])
        data = data[data['base_video'].isin(all_selected_videos)].copy()
    
    print(f"\n{'='*70}")
    print(f"Dataset Overview")
    print(f"{'='*70}")
    
    # total_videos = sum(len(videos) for videos in category_video_info.values())
    # total_waggles = sum(sum(v['n_waggles'] for v in videos) for videos in category_video_info.values())
    # total_clips = len(data)
    total_videos = data['base_video'].nunique()
    total_waggles = data['waggle_uid'].nunique() if 'waggle_uid' in data.columns else 0
    total_clips = len(data)

    print(f"\n{'='*70}")
    print(f"Dataset Overview")
    print(f"{'='*70}")
    print(f"Total unique base videos: {total_videos}")
    print(f"Total waggle runs: {total_waggles}")
    print(f"Total clips: {total_clips}")
    
    print(f"\nCategory Distribution:")
    for cat in sorted(category_video_info.keys()):
        cat_df = data[data['category'] == cat]
        n_videos_cat = cat_df['base_video'].nunique()
        n_waggles_cat = cat_df['waggle_uid'].nunique() if 'waggle_uid' in cat_df.columns else 0
        n_clips_cat = len(cat_df)
        print(f"  {cat:10s}: {n_videos_cat:3d} videos, {n_waggles_cat:4d} waggle runs, {n_clips_cat:5d} clips")
    # Greedy algorithm to split videos while maintaining waggle run ratio
    train_videos = set()
    val_videos = set()
    
    for category, videos_info in category_video_info.items():
        # Sort videos by waggle count (descending) for better bin-packing
        videos_info = sorted(videos_info, key=lambda x: x['n_waggles'], reverse=True)
        
        # Calculate target waggle counts for this category
        total_cat_waggles = sum(v['n_waggles'] for v in videos_info)
        target_train_waggles = int(train_ratio * total_cat_waggles)
        
        # Greedy allocation: add videos to training until we reach target
        current_train_waggles = 0
        cat_train_videos = []
        cat_val_videos = []
        
        for video_info in videos_info:
            # Add to training if we haven't reached target and minimum videos requirement
            if (current_train_waggles < target_train_waggles or 
                len(cat_train_videos) < min_category_train_videos):
                cat_train_videos.append(video_info['video'])
                current_train_waggles += video_info['n_waggles']
            else:
                cat_val_videos.append(video_info['video'])
        
        # Ensure at least one video in validation if we have more than min_category_train_videos
        if len(cat_val_videos) == 0 and len(videos_info) > min_category_train_videos:
            # Move smallest waggle video to validation
            moved_video = cat_train_videos.pop()
            cat_val_videos.append(moved_video)
        
        train_videos.update(cat_train_videos)
        val_videos.update(cat_val_videos)
    
    # Split data
    train_df = data[data['base_video'].isin(train_videos)].reset_index(drop=True)
    val_df = data[data['base_video'].isin(val_videos)].reset_index(drop=True)
    
    # Verify no leakage
    train_base = set(train_df['base_video'].unique())
    val_base = set(val_df['base_video'].unique())
    assert len(train_base & val_base) == 0, "Data leakage detected!"
    
    # Calculate waggle run statistics
    train_waggles = train_df['waggle_uid'].nunique()
    val_waggles = val_df['waggle_uid'].nunique()
    total_waggles_actual = train_waggles + val_waggles
    actual_train_ratio = train_waggles / total_waggles_actual if total_waggles_actual > 0 else 0
    
    # Verify validation set is not empty
    if len(val_df) == 0:
        raise ValueError(
            "Validation set is empty! This can happen with very small subsets. "
            "Try increasing subset_size, subset_videos, or subset_per_category values."
        )
    
    # Print split summary
    print(f"\n{'='*70}")
    print(f"Split Summary (target waggle ratio={train_ratio:.2%})")
    print(f"{'='*70}")
    print(f"Train: {len(train_videos):3d} videos, {train_waggles:4d} waggle runs ({actual_train_ratio:.2%}), {len(train_df):5d} clips")
    print(f"Val:   {len(val_videos):3d} videos, {val_waggles:4d} waggle runs ({1-actual_train_ratio:.2%}), {len(val_df):5d} clips")
    
    print(f"\nPer-Category Split:")
    for cat in sorted(category_video_info.keys()):
        train_cat = train_df[train_df['category'] == cat]
        val_cat = val_df[val_df['category'] == cat]
        
        train_cat_waggles = train_cat['waggle_uid'].nunique()
        val_cat_waggles = val_cat['waggle_uid'].nunique()
        total_cat_waggles = train_cat_waggles + val_cat_waggles
        cat_ratio = train_cat_waggles / total_cat_waggles if total_cat_waggles > 0 else 0
        
        print(f"  {cat:10s}:", end="")
        print(f" Train: {len(train_cat['base_video'].unique()):2d} videos, "
              f"{train_cat_waggles:3d} waggles ({cat_ratio:5.1%}), {len(train_cat):4d} clips", end="")
        print(f" | Val: {len(val_cat['base_video'].unique()):2d} videos, "
              f"{val_cat_waggles:3d} waggles ({1-cat_ratio:5.1%}), {len(val_cat):4d} clips")
    
    # Check for potential issues
    print(f"\n{'='*70}")
    print(f"Validation Checks")
    print(f"{'='*70}")
    
    issues = []
    
    # Check ratio deviation
    ratio_deviation = abs(actual_train_ratio - train_ratio)
    if ratio_deviation > 0.1:
        issues.append(
            f"⚠️  Waggle run ratio deviation: {ratio_deviation:.1%} "
            f"(target: {train_ratio:.1%}, actual: {actual_train_ratio:.1%})"
        )
    
    for cat in sorted(category_video_info.keys()):
        train_cat = train_df[train_df['category'] == cat]
        val_cat = val_df[val_df['category'] == cat]
        
        if len(train_cat) == 0:
            issues.append(f"⚠️  Category '{cat}' missing from training set!")
        
        train_waggles_cat = train_cat['waggle_run_id'].nunique()
        val_waggles_cat = val_cat['waggle_run_id'].nunique()
        
        if val_waggles_cat > train_waggles_cat:
            issues.append(
                f"⚠️  Category '{cat}': Val waggles ({val_waggles_cat}) "
                f"> Train waggles ({train_waggles_cat})"
            )
    
    if issues:
        print("Issues detected:")
        for issue in issues:
            print(f"  {issue}")
        print("\n💡 Consider adjusting train_ratio or dataset size")
    else:
        print("✓ All checks passed!")
        print("  - No data leakage")
        print("  - All categories represented in training")
        print(f"  - Waggle run ratio within ±10% of target ({train_ratio:.0%})")
    
    return train_df, val_df


def create_video_based_folds(
    data: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    video_col: str = 'video_name',
    subset_size: float = None,
    subset_videos: int = None,
    subset_per_category: Dict[str, int] = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create K-fold splits balancing waggle runs across folds.
    
    Args:
        data: DataFrame with annotations (must have 'waggle_run_id' column)
        n_splits: Number of folds
        seed: Random seed
        video_col: Column name containing video names
        subset_size: Use only this fraction of total videos
        subset_videos: Use only this many total videos
        subset_per_category: Dict mapping category to number of videos
        
    Returns:
        List of (train_df, val_df) tuples
    """
    np.random.seed(seed)
    
    if 'waggle_run_id' not in data.columns:
        raise ValueError("Data must contain 'waggle_run_id' column")
    
    # Add base video and category columns
    data = data.copy()
    data['base_video'] = data[video_col].apply(get_base_video_name)
    data['category'] = data['base_video'].apply(get_video_category)
    
    # Get video statistics
    video_stats = get_fold_statistics(data, video_col)
    
    # Group videos by category with their waggle counts
    category_video_info = defaultdict(list)
    for _, row in video_stats.iterrows():
        category_video_info[row['category']].append({
            'video': row['base_video'],
            'n_waggles': row.get('n_waggle_runs', 0)
        })
    
    # Apply subset selection (similar to split function)
    # ... [subset selection code - same as in split_by_waggle_runs]
    
    print(f"\n{'='*70}")
    print(f"Waggle-Balanced {n_splits}-Fold Cross-Validation Setup")
    print(f"{'='*70}")
    
    # For each category, assign videos to folds trying to balance waggle runs
    video_to_fold = {}
    
    for category, videos_info in category_video_info.items():
        # Sort by waggle count for better distribution
        videos_info = sorted(videos_info, key=lambda x: x['n_waggles'], reverse=True)
        
        # Initialize fold waggle counts
        fold_waggles = [0] * n_splits
        
        # Greedy assignment: assign each video to fold with fewest waggles
        for video_info in videos_info:
            # Find fold with minimum waggles
            min_fold = np.argmin(fold_waggles)
            video_to_fold[video_info['video']] = min_fold
            fold_waggles[min_fold] += video_info['n_waggles']
        
        print(f"Category {category:10s}: {len(videos_info):3d} videos, "
              f"waggles per fold: {fold_waggles}")
    
    # Create fold splits
    fold_splits = []
    
    for fold_idx in range(n_splits):
        val_videos = {v for v, f in video_to_fold.items() if f == fold_idx}
        train_videos = {v for v, f in video_to_fold.items() if f != fold_idx}
        
        train_df = data[data['base_video'].isin(train_videos)].reset_index(drop=True)
        val_df = data[data['base_video'].isin(val_videos)].reset_index(drop=True)
        
        # Verify no leakage
        assert len(train_videos & val_videos) == 0, f"Fold {fold_idx}: Leakage detected!"
        
        train_waggles = train_df['waggle_run_id'].nunique()
        val_waggles = val_df['waggle_run_id'].nunique()
        
        print(f"\nFold {fold_idx + 1}/{n_splits}:")
        print(f"  Train: {len(train_videos):3d} videos, {train_waggles:4d} waggles, {len(train_df):5d} clips")
        print(f"  Val:   {len(val_videos):3d} videos, {val_waggles:4d} waggles, {len(val_df):5d} clips")
        
        fold_splits.append((train_df, val_df))
    
    return fold_splits


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    video_column: str = 'base_video'
) -> bool:
    """
    Verify no video appears in both train and val sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        video_column: Column name containing video identifiers
        
    Returns:
        True if no leakage, raises AssertionError otherwise
    """
    train_videos = set(train_df[video_column].unique())
    val_videos = set(val_df[video_column].unique())
    
    overlap = train_videos & val_videos
    
    if len(overlap) > 0:
        raise AssertionError(
            f"Data leakage detected! {len(overlap)} videos appear in both sets:\n"
            f"{list(overlap)[:5]}{'...' if len(overlap) > 5 else ''}"
        )
    
    return True