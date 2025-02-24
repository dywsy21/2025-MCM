model establishment 部分
建立的模型需要用于解决以下问题：
Develop a model for medal counts for each country (for Gold and total medals at a minimum). Include estimates of the uncertainty/precision of your model predictions and measures of how well model performs.
o Based on your model, what are your projections for the medal table in the Los Angeles, USA summer Olympics in 2028? Include prediction intervals for all results. Which countries do you believe are most likely to improve? Which will do worse than in 2024?

本组的想法如下：
我们将使用回归模型来预测奖牌数。我们将根据往期获奖情况评估一个国家的体育实力，并将其划分为5个Tier(align with:Task1-subtask3)。我们将分别使用金牌、银牌和铜牌数作为目标变量，使用历史奖牌数、主办国家等因素作为解释变量，,为每个Tier训练各自的模型，使用训练集、验证集和测试集来优化模型的性能，使用 MAE       MAPE       MSE       RMSE       R2 来评估模型的性能。我们还将计算预测的不确定性/精度，并给出其预测区间。具体的流程如下：
1. 选取一个tier中的所有国家的最近8届奥运赛事的主办国家和金银铜牌数作为数据集进行训练，以Tier 1的金牌数模型训练为例，具体步骤如下：
    - 1.1 数据加载：加载数据集，包括主办国家、历年金牌数、银牌数、铜牌数等字段
    - 1.2 数据转换：对主办国家采用one-hot编码，SMOTE随机过采样，根据不平衡问题的技术，通过生成新的合成样本来增加少数类样本的数量，从而平衡数据集中的类别分布
    - 1.3 特征工程：通过以下算法（附图）提取特征，把特征选择变成 0-1变量，寻找最优的特征组合， 0代表不选择这个特征， 1代表选择的特征，使用无优化的随机森林模型多次训练，看选择哪种特征组合会得到最佳的效果,在Tier 1的金牌数的训练过程中，SSA麻雀算法表现最优（附图）
    - 1.4 数据划分：由于数据量相对较小，采用交叉验证中的常用方法，即kfold验证，将数据集划分为训练集、验证集和测试集
    - 1.5 模型训练：使用训练集，利用不同模型（LSBoost、XGBoost、MultiVariate Linear、Gaussian、decision tree、random forest）训练模型 （附图）
    - 1.6 模型评估：使用验证集调整超参数，决定哪组超参数具有更好的性能
    - 1.6 模型评估：使用测试集评估模型性能，计算MAE、MAPE、MSE、RMSE、R2等指标（附图附表）
    - 1.7 模型选择：根据评估结果选择最优模型，具体计算方式：TOPSIS法（原因：MAE、MAPE、MSE、RMSE越小越好，R2越大（越接近1越好）），赋初权均为1，在Tier 1的金牌数中，XGBOOST表现最优
    - 1.8 模型优化：使用不同优化算法（SSA \cite{8}, DBO \cite{9}, SCA \cite{10}, SA \cite{11}, PSO \cite{12}, SO \cite{13}, POA \cite{14},  
GWO \cite{15}, IGWO \cite{16}, AVOA \cite{17}, CSA \cite{18}, GTO \cite{19}, NGO \cite{20},  
WSO \cite{21}, CGO \cite{22}, INFO \cite{23}, COA \cite{24}, RIME \cite{25}, KOA \cite{26}, RUN \cite{27}.）对模型进行优化，重复1.6-1.7步骤，选取其中的最佳优化算法，在Tier 1的金牌数中，SSA（sparrow search algorithm）表现最优
    - 1.9 衡量不确定性：使用验证集进行模型预测，计算预测的不确定性/精度，给出预测区间，方法：Gaussian Probability Interval Prediction