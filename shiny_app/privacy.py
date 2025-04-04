
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from itertools import combinations

matplotlib.use('Qt5Agg')

# Load the synthetic user data
data = pd.read_csv("data/synthetic_user_data.csv")
# data = pd.read_csv("shiny_app/data/synthetic_user_data.csv") # manual code execution

users = data.drop_duplicates(subset=['user_id']) # create df based on data with unique user_ids

# replace nan by 'unknown'
users = users.fillna('unknown')

# Define the quasi-identifier columns
quasi_identifiers = ["age", "gender", "location", "persona",  "listen_freq"]

# Define the sensitive attribute column
sensitive_attributes = ["like", "duration_listened", "preferred_genre"]

# number of unique values of 'age'

unique_ages = users['age'].nunique()


##################################
# K-anonymity Calculation
##################################
# calculate k-anonymity for all possible combinations of quasi_identifiers
def compute_k_anonymity(df, quasi_identifiers):
    eq_classes = df.groupby(quasi_identifiers)
    return eq_classes.size().min()

k_anonymity_dict = {}
for i in range(1, len(quasi_identifiers) + 1):
    for combo in combinations(quasi_identifiers, i):
        k_anonymity_dict[combo] = compute_k_anonymity(users, list(combo))

print("K-anonymity per quasi-identifier combination:")
for combo, k in k_anonymity_dict.items():
    print(combo, k)

# create age bins based on age distribution and append to users dataframe

# create age bins based on age distribution and append to users dataframe
users.loc[:,'age_bins'] = pd.cut(users['age'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20', '21-40', '41-60', '61-80', '81-100'])
users.loc[:,'age_bins'] = users['age_bins'].cat.remove_unused_categories().astype(str)
users['age_bins'] = users['age_bins'].astype(str)

# recalculate k-anonymity based on age_bins
quasi_identifiers = ["age_bins", "gender", "location", "persona", "listen_freq"]

# calculate k-anonymity for all possible combinations of quasi_identifiers
k_anonymity_dict = {}
for i in range(1, len(quasi_identifiers) + 1):
    for combo in combinations(quasi_identifiers, i):
        k_anonymity_dict[combo] = compute_k_anonymity(users, list(combo))

print("K-anonymity per quasi-identifier combination:")
for combo, k in k_anonymity_dict.items():
    print(combo, k)

##################################
# L-diversity Calculation for Multiple Sensitive Attributes
##################################

# Function to check l-Diversity
def check_l_diversity(data, group_column, sensitive_column, l=2):
    grouped = data.groupby(group_column)[sensitive_column].apply(lambda x: len(set(x)))
    return grouped >= l

# Apply l-Diversity check (minimum 2 genres in each age group)
for sensitive_attribute in sensitive_attributes:
    print(check_l_diversity(users, "age_bins", sensitive_attribute, l=2))


##################################
# T-closeness Calculation for Multiple Sensitive Attributes
##################################
def compute_t_closeness(df, quasi_identifiers, sensitive_attr):
    """
    Compute the t-closeness metric for a given sensitive attribute.
    This implementation uses total variation distance between the
    overall distribution and the distribution in each equivalence class.
    """
    # Overall distribution of the sensitive attribute
    overall_dist = df[sensitive_attr].value_counts(normalize=True)

    eq_classes = df.groupby(quasi_identifiers)
    t_values = []

    for _, group in eq_classes:
        group_dist = group[sensitive_attr].value_counts(normalize=True)
        # Use the union of categories from overall and group distributions
        categories = overall_dist.index.union(group_dist.index)
        p = overall_dist.reindex(categories, fill_value=0)
        q = group_dist.reindex(categories, fill_value=0)
        # Total variation distance: 0.5 * sum(|p - q|)
        tv_distance = 0.5 * np.sum(np.abs(p - q))
        t_values.append(tv_distance)

    # The t-closeness value is the maximum TV distance among all equivalence classes.
    return max(t_values)


quasi_identifiers = ["age_bins"]
sensitive_attributes = ["like", "duration_listened", "preferred_genre"]

t_closeness_dict = {}

for sensitive_attr in sensitive_attributes:
    df_copy = users.copy()

    # Handle specific sensitive attribute transformations
    if sensitive_attr == "like":
        # Convert -1/0/1 to categorical labels
        like_mapping = {-1: 'No', 0: 'Neutral', 1: 'Yes'}
        df_copy[sensitive_attr] = df_copy[sensitive_attr].map(like_mapping).astype('category')

    elif np.issubdtype(df_copy[sensitive_attr].dtype, np.number):
        # Bin continuous numeric attributes (e.g., duration_listened)
        df_copy[sensitive_attr] = pd.qcut(df_copy[sensitive_attr], q=5, duplicates='drop')

    # Compute t-closeness
    t_closeness_dict[sensitive_attr] = compute_t_closeness(df_copy, quasi_identifiers, sensitive_attr)

print("T-closeness per sensitive attribute:", t_closeness_dict)

# Calculate t-closeness for preferred_genre_cluster

sensitive_attributes = ["like", "duration_listened", "preferred_genre_cluster"]

t_closeness_dict = {}

for sensitive_attr in sensitive_attributes:
    df_copy = users.copy()

    # Handle specific sensitive attribute transformations
    if sensitive_attr == "like":
        # Convert -1/0/1 to categorical labels
        like_mapping = {-1: 'No', 0: 'Neutral', 1: 'Yes'}
        df_copy[sensitive_attr] = df_copy[sensitive_attr].map(like_mapping).astype('category')

    elif np.issubdtype(df_copy[sensitive_attr].dtype, np.number):
        # Bin continuous numeric attributes (e.g., duration_listened)
        df_copy[sensitive_attr] = pd.qcut(df_copy[sensitive_attr], q=5, duplicates='drop')

    # Compute t-closeness
    t_closeness_dict[sensitive_attr] = compute_t_closeness(df_copy, quasi_identifiers, sensitive_attr)

print("T-closeness per sensitive attribute:", t_closeness_dict)


# plot total distribution of sensitive attribute preferred_genre_cluster and per age_bins
def plot_group_distributions(df, target_col, groupby_col):
    """
    Plot normalized distributions of a categorical target_col
    across groups defined by groupby_col in one overlaid plot.
    """
    # Compute normalized value counts per group
    grouped = df.groupby(groupby_col)[target_col].value_counts(normalize=True).unstack(fill_value=0)

    # Plot
    plt.figure(figsize=(10, 6))
    for group_name, row in grouped.iterrows():
        plt.plot(row.index, row.values, marker='o', label=str(group_name))

    plt.title(f"Distribution of '{target_col}' across '{groupby_col}' groups")
    plt.xlabel(target_col)
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    plt.legend(title=groupby_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot the distribution of the sensitive attribute "preferred_genre_cluster" by "age_bins"
plot_group_distributions(users, "preferred_genre_cluster", ["age_bins"])

# Apply  central differential privacy to sensitive attribute to duration_listened
def apply_laplace_mechanism(series, epsilon, sensitivity):
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=len(series))
    return series + noise

# Privacy parameters
epsilon = 1
sensitivity = data['duration_listened'].max() - data['duration_listened'].min()

# Apply Laplace noise
noisy_values = apply_laplace_mechanism(data['duration_listened'], epsilon, sensitivity)

# Clip per-row: DP value should not exceed duration and be non-negative
data['duration_listened_dp'] = np.clip(noisy_values, a_min=0, a_max=data['duration_ms']).astype(int)

# Compute RMSE and normalized RMSE by range

original = data['duration_listened']
dp = data['duration_listened_dp']

# Compute RMSE
rmse = np.sqrt(mean_squared_error(original, dp))

# Compute standardization
range_val = original.max() - original.min()


nrmse_range = rmse / range_val

print(f"RMSE: {rmse:.4f}")
print(f"NRMSE (range): {nrmse_range:.4f}")


if __name__ == "__main__":
    # Run the code
    pass