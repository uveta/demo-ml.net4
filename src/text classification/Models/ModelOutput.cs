using Microsoft.ML.Data;

namespace text_classification.Models;

public class ModelOutput : ModelInput
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }

    public float Probability { get; set; }
    public float Score { get; set; }
}