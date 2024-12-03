using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.Transforms.Text;
using named_entity_recognition.Models;
using TorchSharp;

const int BatchSize = 32;
const int MaxEpochs = 2;

try
{
    var allowedLevels = new HashSet<ChannelMessageKind>
    {
        /*ChannelMessageKind.Trace,*/
        ChannelMessageKind.Info, ChannelMessageKind.Warning, ChannelMessageKind.Error
    };
    var context = new MLContext { FallbackToCpu = false };
    context.Log += (_, e) =>
    {
        if (!allowedLevels.Contains(e.Kind)) return;
        Console.WriteLine($"[{e.Kind:G}] {e.Message}");
    };
    // Check for available GPU devices
    int gpuDevices = torch.cuda.device_count();
    if (gpuDevices > 0)
    {
        // Select the first GPU device
        context.GpuDeviceId = 0;
    }
    else
    {
        Console.WriteLine("No GPU devices found. Falling back to CPU.");
        context.FallbackToCpu = true;
    }

    // train
    var labels = context.Data.LoadFromEnumerable(
    [
        new Label { Key = "PERSON" },
        new Label { Key = "CITY" },
        new Label { Key = "COUNTRY" }
    ]);
    var trainingData = context.Data.LoadFromEnumerable(GetMockData());
    var estimator = context.Transforms.Text.NormalizeText(
            inputColumnName: "Sentence",
            outputColumnName: "Sentence",
            caseMode: TextNormalizingEstimator.CaseMode.None,
            keepDiacritics: false,
            keepPunctuations: false,
            keepNumbers: true)
        .Append(
            context.Transforms.Conversion.MapValueToKey(
                outputColumnName: "Label",
                inputColumnName: "Label",
                addKeyValueAnnotationsAsText: false,
                keyData: labels))
        .Append(
            context.MulticlassClassification.Trainers.NamedEntityRecognition(
                labelColumnName: "Label",
                outputColumnName: "PredictedLabel",
                sentence1ColumnName: "Sentence",
                batchSize: BatchSize,
                maxEpochs: MaxEpochs))
        .Append(
            context.Transforms.Conversion.MapKeyToValue(
                outputColumnName: "PredictedLabel",
                inputColumnName: "PredictedLabel"));
    var transformer = estimator.Fit(trainingData);
    var engine = context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(transformer);

    // test
    var testData = new ModelInput { Sentence = "Mark and John live in Canada with a cat" };
    var prediction = engine.Predict(testData);
    Console.WriteLine($"\n\nPredicted labels Label: {string.Join(", ", prediction.PredictedLabel)}\n\n");
    transformer.Dispose();

    Console.WriteLine("Success!");
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
return;

static List<ModelInput> GetMockData()
{
    var trainingData = new List<ModelInput>();
    for (var i = 0; i < 1500; i++)
    {
        trainingData.Add(
            new ModelInput
            {
                Sentence = "Alice and Bob live in London with a dog",
                Label = ["PERSON", "0", "PERSON", "0", "0", "CITY", "0", "0", "0"]
            });
    }
    return trainingData;
}
