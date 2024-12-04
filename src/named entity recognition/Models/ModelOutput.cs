using Microsoft.ML.Data;

namespace named_entity_recognition.Models;

internal class ModelOutput
{
    [ColumnName("Sentence")]
    public string Sentence { get; set; } = string.Empty;

    [ColumnName("Label")]
    public uint[] Label { get; set; } = [];

    [ColumnName("PredictedLabel")]
    public string[] PredictedLabel { get; set; } = [];
}
