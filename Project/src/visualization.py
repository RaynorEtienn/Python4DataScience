import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.dates as mdates


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
