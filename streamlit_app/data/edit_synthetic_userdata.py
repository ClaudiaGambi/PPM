

import pandas as pd
import numpy as np

# Load synthetic user data
user_data = pd.read_csv("synthetic_user_data.csv")

def edit_synthetic_userdata(user_data, destination_user_id, source_user_id, max_fraction=0.5):
    """
    Copies a random fraction (between 0 and max_fraction) of full rows from `source_user_id`
    and assigns them to `destination_user_id` without removing existing rows of `destination_user_id`.

    The actual fraction is randomly chosen between 0 and max_fraction, ensuring variability.
    """
    # Get all rows belonging to the source user
    source_user_rows = user_data[user_data['user_id'] == source_user_id]

    if source_user_rows.empty:
        return user_data  # No data to copy

    # Randomly determine the fraction to copy (between 0 and max_fraction)
    actual_fraction = np.random.uniform(0, max_fraction)

    # Determine the number of rows to copy
    num_rows_to_copy = int(len(source_user_rows) * actual_fraction)
    num_rows_to_copy = max(0, min(num_rows_to_copy, len(source_user_rows)))  # Ensure valid range

    if num_rows_to_copy == 0:
        return user_data  # No rows selected

    # Randomly select rows
    copied_rows = source_user_rows.sample(n=num_rows_to_copy, replace=False)

    # Change the user_id in the copied rows to the destination user
    copied_rows = copied_rows.copy()  # Avoid modifying original DataFrame
    copied_rows['user_id'] = destination_user_id

    # Remove potential duplicates (if destination_user_id already has these exact rows)
    existing_rows = user_data[user_data['user_id'] == destination_user_id]
    copied_rows = copied_rows[~copied_rows.apply(tuple, axis=1).isin(existing_rows.apply(tuple, axis=1))]

    if copied_rows.empty:
        return user_data  # No new rows to add

    updated_user_data = pd.concat([user_data, copied_rows], ignore_index=True)

    return updated_user_data

# Example usage
user_data = edit_synthetic_userdata(user_data, destination_user_id=1, source_user_id=9, max_fraction=0.8)

def print_identical_rows(user_data, source_user_id, destination_user_id):
    """
    Prints identical unique rows between `source_user_id` and `destination_user_id`.
    """
    # Get all rows belonging to both users
    source_rows = user_data[user_data['user_id'] == source_user_id]
    destination_rows = user_data[user_data['user_id'] == destination_user_id]

    # Ensure there are common columns
    common_columns = [col for col in user_data.columns if col != 'user_id']

    if not common_columns:
        print("No comparable columns found.")
        return

    # Find identical rows based on all columns except 'user_id'
    identical_rows = pd.merge(source_rows[common_columns], destination_rows[common_columns], how='inner')

    if identical_rows.empty:
        print(f"No identical rows found between user {source_user_id} and user {destination_user_id}.")
    else:
        print(f"Identical unique rows between user {source_user_id} and user {destination_user_id}:")
        print(identical_rows)

# Example usage
print_identical_rows(user_data, source_user_id=14, destination_user_id=13)


# Save the modified data
user_data.to_csv('synthetic_user_data_test.csv', index=False)






