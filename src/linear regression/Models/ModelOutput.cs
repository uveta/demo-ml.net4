using Microsoft.ML.Data;

namespace linear_regression.Models;

public class ModelOutput
{
    [ColumnName("Score")]
    public float Price { get; set; }
}
