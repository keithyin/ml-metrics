def compute_auuc2(df, reward_col_name, treatment_col_name, model_pred_tau_col_name):
    
    model_names = [model_pred_tau_col_name]
    random_cols = []
    
    sample_size = df.shape[0]
    percentile_span = int(sample_size / 10)
    
    for i in range(50):
        random_col = "__random_{}__".format(i)
        df[random_col] = np.random.rand(df.shape[0])
        model_names.append(random_col)
        random_cols.append(random_col)
    
    
    lift_rewards = []
    
    for model_name in model_names:
        
        sorted_df = df.sort_values(by=model_name, ascending=False).reset_index()
        sorted_df.index = sorted_df.index + 1
        
        sorted_df["cumsum_treat"] = (sorted_df[treatment_col_name].cumsum() + 1e-6)
        sorted_df["cumsum_ctl"] = (sorted_df.index.values - sorted_df[treatment_col_name].cumsum() + 1e-6)
        
        sorted_df["cumsum_reward_treat"] = (sorted_df[reward_col_name] * sorted_df[treatment_col_name]).cumsum()
        
        sorted_df["cumsum_reward_ctl"] = (sorted_df[reward_col_name] * (1 - sorted_df[treatment_col_name])).cumsum()
        
        lift_reward = sorted_df["cumsum_reward_treat"] / sorted_df["cumsum_treat"] - sorted_df["cumsum_reward_ctl"] / sorted_df["cumsum_ctl"]
        
        
        # ate * (# teatment population)
        gain_reward = lift_reward * sorted_df.index
        gain_reward = gain_reward / gain_reward.iloc[-1]
        
        if model_name == model_pred_tau_col_name:
            percentile_exp_ctl_info= []

            for i in range(1, 11):
                if i < 10:
                    idx = i * percentile_span - 1
                else:
                    idx = sample_size
                exp_num = int(sorted_df["cumsum_treat"].loc[idx])
                ctl_num = int(sorted_df["cumsum_ctl"].loc[idx])
                exp_ratio = float(exp_num) / (exp_num + ctl_num)
                percentile_exp_ctl_info.append("{}%->\t exp:{},\t ctl:{},\t exp_ratio:{:.2f}%".format(i * 10, exp_num, ctl_num, exp_ratio * 100))

            print("\n".join(percentile_exp_ctl_info))
        
        lift_rewards.append(gain_reward)
            
    lift_rewards = pd.concat(lift_rewards, join="inner", axis=1)
    
    lift_rewards.loc[0] = np.zeros((lift_rewards.shape[1],))
    
    lift_rewards = lift_rewards.sort_index().interpolate()
    
    lift_rewards.columns = model_names
    
    lift_rewards["RANDOM__"] = lift_rewards[random_cols].mean(axis=1)
    
    lift_rewards.drop(random_cols, axis=1, inplace=True)
    print(lift_rewards.mean(axis=0))
    lift_rewards.plot()
    return lift_rewards
  
  
  
  def compute_aucc(df, cost_col_name, reward_col_name, treatment_col_name, model_pred_tau_col_name):
    
    model_names = [model_pred_tau_col_name]
    random_cols = []
    
    for i in range(50):
        random_col = "__random_{}__".format(i)
        df[random_col] = np.random.rand(df.shape[0])
        model_names.append(random_col)
        random_cols.append(random_col)
    
    sample_size = df.shape[0]
    percentile_span = int(sample_size / 10)
    
    lift_rewards = []
    lift_costs = []
    
    for model_name in model_names:
        
        sorted_df = df.sort_values(by=model_name, ascending=False).reset_index()
        sorted_df.index = sorted_df.index + 1
        
        sorted_df["cumsum_treat"] = (sorted_df[treatment_col_name].cumsum() + 1e-6)
        sorted_df["cumsum_ctl"] = (sorted_df.index.values - sorted_df[treatment_col_name].cumsum() + 1e-6)
        
        sorted_df["cumsum_reward_treat"] = (sorted_df[reward_col_name] * sorted_df[treatment_col_name]).cumsum()
        sorted_df["cumsum_cost_treat"] = (sorted_df[cost_col_name] * sorted_df[treatment_col_name]).cumsum()
        
        sorted_df["cumsum_reward_ctl"] = (sorted_df[reward_col_name] * (1 - sorted_df[treatment_col_name])).cumsum()
        sorted_df["cumsum_cost_ctl"] = (sorted_df[cost_col_name] * (1 - sorted_df[treatment_col_name])).cumsum()
        
        lift_reward = sorted_df["cumsum_reward_treat"] / sorted_df["cumsum_treat"] - sorted_df["cumsum_reward_ctl"] / sorted_df["cumsum_ctl"]
        
        lift_cost = sorted_df["cumsum_cost_treat"] / sorted_df["cumsum_treat"] - sorted_df["cumsum_cost_ctl"] / sorted_df["cumsum_ctl"]
        
        # ate * (# teatment population)
        gain_reward = sorted_df["cumsum_treat"] * lift_reward
        gain_cost = sorted_df["cumsum_treat"] * lift_cost
        
        if model_name == model_pred_tau_col_name:
            
            percentile_exp_ctl_info= []
            
            for i in range(1, 11):
                if i < 10:
                    idx = i * percentile_span - 1
                else:
                    idx = sample_size
                exp_num = int(sorted_df["cumsum_treat"].loc[idx])
                ctl_num = int(sorted_df["cumsum_ctl"].loc[idx])
                exp_ratio = float(exp_num) / (exp_num + ctl_num)
                percentile_exp_ctl_info.append("{}%->\t exp:{},\t ctl:{},\t exp_ratio:{:.2f}%".format(i * 10, exp_num, ctl_num, exp_ratio * 100))
            
            print("\n".join(percentile_exp_ctl_info))
                
        
        lift_rewards.append(gain_reward)
        
        lift_costs.append(gain_cost)
    
    lift_rewards = pd.concat(lift_rewards, join="inner", axis=1)
    lift_cost = pd.concat(lift_costs, join="inner", axis=1)
    
    lift_rewards.loc[0] = np.zeros((lift_rewards.shape[1],))
    lift_cost.loc[0] = np.zeros((lift_cost.shape[1],))
    
    lift_rewards = lift_rewards.sort_index().interpolate()
    lift_cost = lift_cost.sort_index().interpolate()
    
    lift_rewards.columns = model_names
    lift_cost.columns = model_names
    
    lift_rewards["RANDOM__"] = lift_rewards[random_cols].mean(axis=1)
    lift_cost["RANDOM__"] = lift_cost[random_cols].mean(axis=1)
    
    lift_rewards.drop(random_cols, axis=1, inplace=True)
    lift_cost.drop(random_cols, axis=1, inplace=True)
    
    return lift_rewards, lift_cost
