import numpy as np

from scripts.test import evaluate_model

if __name__ == "__main__":
    _, hidden_states = evaluate_model(data_dir="../data/240_rule/df_test_combine.csv",
                                      model_path="../models/240_rule/4_17_20_lstm_layers_3_hidden_1024_input_10.h5",
                                      results_dir="../results")
    all_hidden_states = np.vstack(hidden_states)
    print(all_hidden_states.shape)
