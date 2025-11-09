import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# TP2: K-Nearest Neighbors (KNN) - Updated and cleaned version
# This script contains small security-related KNN exercises:
#  - Manual 2D classification (network traffic)
#  - Manual 1D classification (login attempts)
#  - Small synthetic network intrusion dataset
#  - Simple email spam detection example with plot saved
# ============================================================


def part1_manual_2d():
    print("\n" + "=" * 60)
    print("PART 1: Manual KNN for 2D Network Traffic Classification (cleaned texts)")
    print("=" * 60)

    # Training data (packet features or similar)
    X_train_2d = np.array([[1.0, 0.25],
                           [0.4, 0.10],
                           [0.5, 0.50],
                           [1.0, 1.0]])
    y_train_2d = np.array([0, 0, 1, 1])  # 0=Normal, 1=Attack

    # New point to classify
    P = np.array([0.5, 0.15])

    print(f"New point to classify: ({P[0]}, {P[1]})\n")

    # Euclidean distances
    distances_2d = np.sqrt(np.sum((X_train_2d - P) ** 2, axis=1))
    for i, (point, label, dist) in enumerate(zip(X_train_2d, y_train_2d, distances_2d)):
        label_str = "Normal" if label == 0 else "Attack"
        print(f"Point {i + 1}: {point} | Label: {label_str} | Distance: {dist:.4f}")

    sorted_indices = np.argsort(distances_2d)
    print(f"\nSorted distances: {distances_2d[sorted_indices]}")

    k = 3
    print(f"\nFor K={k}, the {k} nearest neighbors are:")
    for i in range(k):
        idx = sorted_indices[i]
        label_str = "Normal" if y_train_2d[idx] == 0 else "Attack"
        print(f"  - Point {idx + 1}: {X_train_2d[idx]} | Label: {label_str} | Distance: {distances_2d[idx]:.4f}")

    nearest_labels = y_train_2d[sorted_indices[:k]]
    prediction = np.bincount(nearest_labels).argmax()
    print(f"\nMajority vote (K={k}): {nearest_labels}")
    print(f"Predicted class: {'Normal' if prediction == 0 else 'Attack'}")


def part2_manual_1d():
    print("\n" + "=" * 60)
    print("PART 2: Manual KNN for 1D Login Attempt Classification")
    print("=" * 60)

    # Training data: number of failed attempts or time since last login, etc.
    X_train_1d = np.array([[0], [3], [4], [6], [9]])
    y_train_1d = np.array([0, 1, 1, 0, 0])  # 0=Legitimate, 1=Suspicious

    # New login attempt to evaluate
    X_test_1d = np.array([[5.5]])
    y_true_1d = np.array([1])  # assume the true label is suspicious for this example

    print(f"New login value: {X_test_1d[0][0]}\n")

    # Manhattan distances (1D -> absolute difference)
    distances_1d = np.abs(X_train_1d.flatten() - X_test_1d[0][0])
    data_1d = pd.DataFrame({
        'Login Value': X_train_1d.flatten(),
        'Label': y_train_1d,
        'Distance': distances_1d
    })
    data_1d = data_1d.sort_values('Distance')
    print(data_1d.to_string(index=False))

    print("\nPrediction & MAE for several K values:")
    results = []
    for k in [1, 3, 5]:
        model = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        model.fit(X_train_1d, y_train_1d)
        y_pred = model.predict(X_test_1d)
        mae = mean_absolute_error(y_true_1d, y_pred)
        results.append({'K': k, 'Predicted': int(y_pred[0]), 'MAE': mae})
        print(f"K = {k}, Predicted = {int(y_pred[0])}, MAE = {mae:.0f}")

    best = min(results, key=lambda r: r['MAE'])
    print(f"\nBest K (lowest MAE) = {best['K']} (MAE = {best['MAE']:.0f})")


def part3_synthetic_intrusion():
    print("\n" + "=" * 60)
    print("PART 3: Small Synthetic Network Intrusion Detection Example")
    print("=" * 60)

    data = {
        'packet_size': [200, 450, 300, 700, 120, 1000, 150, 400, 800, 130],
        'connection_time': [30, 50, 25, 80, 10, 100, 15, 45, 90, 12],
        'malicious': [0, 0, 0, 1, 0, 1, 0, 0, 1, 0]  # 0=Normal, 1=Attack
    }
    df = pd.DataFrame(data)
    print('\nDataset:')
    print(df)

    X = df[['packet_size', 'connection_time']]
    y = df['malicious']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    for metric in ['manhattan', 'euclidean']:
        print("\n" + "-" * 60)
        print(f"Testing with {metric.capitalize()} distance metric:")
        print("-" * 60)
        results = []
        for k in [1, 3, 5]:
            model = KNeighborsClassifier(n_neighbors=k, metric=metric)
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            results.append((k, mae, y_pred))
            print(f"K={k}, Predictions: {y_pred}, MAE = {mae:.2f}")

        best_k, best_mae, _ = min(results, key=lambda it: it[1])
        print(f"Best K for {metric} = {best_k} (MAE = {best_mae:.2f})")


def part4_email_spam():
    print("\n" + "=" * 60)
    print("PART 4: Email Spam Detection (toy example) - plot saved")
    print("=" * 60)

    # Training dataset (toy features: number of links, count of spammy words)
    X_train_spam = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [5, 6],
        [0, 0],
        [4, 5]
    ])
    y_train_spam = np.array([0, 0, 1, 1, 1, 0, 1])  # 0=Normal, 1=Spam

    X_test_spam = np.array([[2, 1]])
    y_true_spam = np.array([1])  # assume it's spam for evaluation

    spam_df = pd.DataFrame(X_train_spam, columns=['Links', 'SpamWords'])
    spam_df['Label'] = ['Normal' if l == 0 else 'Spam' for l in y_train_spam]
    print('\nTraining Data:')
    print(spam_df)
    print(f"\nNew email to classify: Links={X_test_spam[0][0]}, SpamWords={X_test_spam[0][1]}")
    print("True label: Spam (for evaluation)\n")

    # Plot training data
    plt.figure(figsize=(8, 5))
    for label in [0, 1]:
        mask = y_train_spam == label
        plt.scatter(X_train_spam[mask, 0], X_train_spam[mask, 1],
                    label='Normal' if label == 0 else 'Spam',
                    s=100, alpha=0.7)

    plt.scatter(X_test_spam[0, 0], X_test_spam[0, 1],
                color='red', marker='X', s=200,
                label='New Email', edgecolors='black', linewidth=1.5)

    plt.xlabel('Number of Links')
    plt.ylabel('Number of Spam Keywords')
    plt.title('Email Spam Detection - Training Data')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save plot next to this file
    folder = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(folder, 'spam_detection_plot.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as '{out_path}'")

    print("\nKNN Predictions for different K values:")
    for k in [1, 3, 5]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_spam, y_train_spam)
        y_pred = knn.predict(X_test_spam)
        mae = mean_absolute_error(y_true_spam, y_pred)
        pred_label = 'Spam' if y_pred[0] == 1 else 'Normal'
        print(f"K = {k}, Predicted = {pred_label} ({y_pred[0]}), MAE = {mae:.0f}")


def answers_and_notes():
    print("\n" + "=" * 60)
    print("ANSWERS AND SHORT NOTES")
    print("=" * 60)

    print("\nPART 3 - Notes:")
    print("1. Increasing K -> smoother decision boundary, less sensitive to noise; may underfit if too large.")
    print("2. Adding noisy samples -> can increase MAE; more representative samples generally help.")
    print("3. Manhattan vs Euclidean -> choice depends on features and distribution; scale features to make distances comparable.")

    print("\nPART 4 - Notes:")
    print("1. Features: Links and spam keywords are simplistic but illustrative.")
    print("2. Correct predictions: compare each K's prediction vs. true label (here we assumed Spam).")
    print("3. Best K -> the one with lowest MAE for the held-out or test example(s).")
    print("4. Why large K can be less accurate -> over-smoothing, includes more distant, potentially irrelevant neighbors.")
    print("5. Attacker evasion ideas: reduce links, obfuscate keywords, use images or mimic legitimate style to evade simple detectors.")


def main():
    print("=" * 60)
    print("TP2: K-Nearest Neighbors (KNN) Exercises - Updated")
    print("=" * 60)

    part1_manual_2d()
    part2_manual_1d()
    part3_synthetic_intrusion()
    part4_email_spam()
    answers_and_notes()

    print("\n" + "=" * 60)
    print("TP2 LAB - UPDATED FILE COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
