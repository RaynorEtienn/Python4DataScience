import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.dates as mdates
import pandas as pd


def plot_churn_distribution(df):
    """Plots the distribution of the churn target variable."""
    plt.figure(figsize=(6, 4))
    sb.countplot(x="churn", data=df)
    plt.title("Distribution of Churn")
    plt.show()


def plot_avg_songs_per_session(df):
    """Plots boxplot of average songs per session for churn vs non-churn events."""
    songs_df = df[df["page"] == "NextSong"]
    songs_per_session = (
        songs_df.groupby(["userId", "sessionId"]).size().reset_index(name="songs_count")
    )
    user_churn_map = df[["userId", "churn"]].drop_duplicates()
    songs_per_session = songs_per_session.merge(user_churn_map, on="userId")
    avg_songs_user = (
        songs_per_session.groupby(["userId", "churn"])["songs_count"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 6))
    sb.boxplot(x="churn", y="songs_count", data=avg_songs_user)
    plt.title("Average Songs Played per Session (Churn vs Non-Churn)")
    plt.ylabel("Avg Songs per Session")
    plt.show()


def plot_error_frequency(df):
    """Plots boxplot of error frequency for churn vs non-churn users."""
    errors_df = df[df["page"] == "Error"]
    user_churn_map = df[["userId", "churn"]].drop_duplicates()
    error_counts = errors_df.groupby("userId").size().reset_index(name="error_count")
    error_counts = error_counts.merge(user_churn_map, on="userId", how="right").fillna(
        0
    )

    plt.figure(figsize=(8, 6))
    sb.boxplot(x="churn", y="error_count", data=error_counts)
    plt.title("Frequency of Errors (Churn vs Non-Churn)")
    plt.show()


def plot_user_journeys(df, user_ids):
    """Plots the user journey (page visits over time) for a list of user IDs."""
    for uid in user_ids:
        user_data = df[df["userId"] == uid].sort_values("ts")

        fig, ax = plt.subplots(figsize=(12, 6))

        non_churn = user_data[user_data["churn"] == 0]
        ax.scatter(
            non_churn["ts"],
            non_churn["page"],
            alpha=0.5,
            s=15,
            c="blue",
            label="Normal",
        )

        churn_window = user_data[user_data["churn"] == 1]
        ax.scatter(
            churn_window["ts"],
            churn_window["page"],
            alpha=0.8,
            s=25,
            c="red",
            label="Pre-Churn (10 days)",
        )

        ax.set_title(f"User Journey: {uid}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Page")
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=45)
        plt.legend(loc="upper left")

        plt.tight_layout()
        plt.show()


def plot_categorical_churn_impact(df, columns):
    """
    Plots the churn rate for each category in the specified columns.
    Aggregates by USER first.
    """
    # Create user-level dataset
    # We take the last value for categorical columns
    agg_dict = {"churn_ts": "max"}
    for col in columns:
        if col in df.columns:
            agg_dict[col] = "last"

    user_df = df.groupby("userId").agg(agg_dict)
    user_df["is_churner"] = user_df["churn_ts"].notna().astype(int)

    for col in columns:
        if col not in user_df.columns:
            continue

        plt.figure(figsize=(10, 5))
        # Calculate churn rate (proportion of users who churned) per category
        churn_rate = (
            user_df.groupby(col)["is_churner"].mean().sort_values(ascending=False)
        )

        sb.barplot(x=churn_rate.index, y=churn_rate.values)
        plt.title(f"Churn Rate by {col} (User Level)")
        plt.ylabel("Proportion of Users who Churned")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_numerical_churn_impact(df, columns):
    """
    Plots boxplots for numerical columns split by churn status.
    Aggregates by USER first (mean).
    """
    agg_dict = {"churn_ts": "max"}
    for col in columns:
        if col in df.columns:
            agg_dict[col] = "mean"

    user_df = df.groupby("userId").agg(agg_dict)
    user_df["is_churner"] = user_df["churn_ts"].notna().astype(int)

    for col in columns:
        if col not in user_df.columns:
            continue
        plt.figure(figsize=(8, 6))
        sb.boxplot(x="is_churner", y=col, data=user_df, showfliers=False)
        plt.title(f"Distribution of Average {col} per User")
        plt.show()


def analyze_location(df):
    """
    Extracts state from location and plots churn rate by state.
    Aggregates by USER first.
    """
    if "location" not in df.columns:
        return

    df = df.copy()
    df["state"] = df["location"].apply(
        lambda x: x.split(",")[-1].strip() if x and "," in x else "Unknown"
    )

    user_df = df.groupby("userId").agg({"state": "last", "churn_ts": "max"})
    user_df["is_churner"] = user_df["churn_ts"].notna().astype(int)

    churn_rate = (
        user_df.groupby("state")["is_churner"].mean().sort_values(ascending=False)
    )

    plt.figure(figsize=(15, 6))
    sb.barplot(x=churn_rate.index, y=churn_rate.values)
    plt.title("Churn Rate by State (User Level)")
    plt.ylabel("Proportion of Users who Churned")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def analyze_user_agent(df):
    """
    Extracts OS/Platform from userAgent and plots churn rate.
    Aggregates by USER first.
    """
    if "userAgent" not in df.columns:
        return

    def get_os(agent):
        if not agent:
            return "Unknown"
        if "Windows" in agent:
            return "Windows"
        if "Macintosh" in agent:
            return "Mac"
        if "iPhone" in agent:
            return "iPhone"
        if "iPad" in agent:
            return "iPad"
        if "Android" in agent:
            return "Android"
        if "Linux" in agent:
            return "Linux"
        return "Other"

    df = df.copy()
    df["os"] = df["userAgent"].apply(str).apply(get_os)

    user_df = df.groupby("userId").agg({"os": "last", "churn_ts": "max"})
    user_df["is_churner"] = user_df["churn_ts"].notna().astype(int)

    churn_rate = user_df.groupby("os")["is_churner"].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sb.barplot(x=churn_rate.index, y=churn_rate.values)
    plt.title("Churn Rate by OS (User Level)")
    plt.ylabel("Proportion of Users who Churned")
    plt.show()


def analyze_page_distribution(df, ignore_pages=[]):
    """
    Compares page visit distribution between churn and non-churn users.
    Aggregates by USER first (Proportion of events).
    """
    if "page" not in df.columns:
        return

    # 1. Count page visits per user
    user_page_counts = df.groupby(["userId", "page"]).size().unstack(fill_value=0)

    # 2. Normalize by total events per user
    user_total_events = df.groupby("userId").size()
    user_page_props = user_page_counts.div(user_total_events, axis=0)

    # 3. Add churn status
    churn_status = df.groupby("userId")["churn_ts"].max().notna().astype(int)
    user_page_props["is_churner"] = churn_status

    # 4. Melt for plotting
    melted = user_page_props.melt(
        id_vars="is_churner", var_name="page", value_name="proportion"
    )

    if ignore_pages:
        melted = melted[~melted["page"].isin(ignore_pages)]

    plt.figure(figsize=(15, 8))
    sb.boxplot(
        x="page", y="proportion", hue="is_churner", data=melted, showfliers=False
    )
    plt.title("Page Visit Proportion per User (Churn vs Non-Churn)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
