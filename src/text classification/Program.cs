using Microsoft.ML;
using text_classification.Models;

// Initialize MLContext
var mlContext = new MLContext();

// Load and prepare data
var trainingData = GetTrainingData();
var trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);

// Define the learning pipeline
var pipeline = mlContext.Transforms.Text.FeaturizeText(
        outputColumnName: "Features",
        inputColumnName: nameof(ModelInput.SentimentText))
    .Append(
        mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(ModelInput.Sentiment),
            featureColumnName: "Features"));

// Train the model
var model = pipeline.Fit(trainingDataView);
var testData = GetTestData();
var testDataView = mlContext.Data.LoadFromEnumerable(testData);
// Evaluate the model
var predictions = model.Transform(testDataView);
var metrics = mlContext.BinaryClassification.Evaluate(
    predictions,
    labelColumnName: nameof(ModelInput.Sentiment));

// Output evaluation metrics
Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
return;

List<ModelInput> GetTrainingData()
{
    return
    [
        new ModelInput { SentimentText = "I love this product!", Sentiment = true },
        new ModelInput { SentimentText = "This is the worst purchase I've ever made.", Sentiment = false },
        new ModelInput { SentimentText = "Absolutely fantastic!", Sentiment = true },
        new ModelInput { SentimentText = "Not worth the money.", Sentiment = false },
        new ModelInput { SentimentText = "I'm very satisfied with this.", Sentiment = true },
        new ModelInput { SentimentText = "Terrible quality.", Sentiment = false },
        new ModelInput { SentimentText = "Exceeded my expectations.", Sentiment = true },
        new ModelInput { SentimentText = "I will never buy this again.", Sentiment = false },
        new ModelInput { SentimentText = "Highly recommend it!", Sentiment = true },
        new ModelInput { SentimentText = "Very disappointed.", Sentiment = false },
        new ModelInput { SentimentText = "Great value for the price.", Sentiment = true },
        new ModelInput { SentimentText = "Horrible experience.", Sentiment = false },
        new ModelInput { SentimentText = "I am extremely happy with this purchase.", Sentiment = true },
        new ModelInput { SentimentText = "Would not recommend.", Sentiment = false },
        new ModelInput { SentimentText = "Five stars!", Sentiment = true },
        new ModelInput { SentimentText = "One star.", Sentiment = false },
        new ModelInput { SentimentText = "Fantastic product!", Sentiment = true },
        new ModelInput { SentimentText = "Very poor quality.", Sentiment = false },
        new ModelInput { SentimentText = "I am very pleased.", Sentiment = true },
        new ModelInput { SentimentText = "Not what I expected.", Sentiment = false },
        new ModelInput { SentimentText = "Excellent!", Sentiment = true },
        new ModelInput { SentimentText = "Terrible.", Sentiment = false },
        new ModelInput { SentimentText = "I love it!", Sentiment = true },
        new ModelInput { SentimentText = "Waste of money.", Sentiment = false },
        new ModelInput { SentimentText = "Highly satisfied.", Sentiment = true },
        new ModelInput { SentimentText = "Very bad.", Sentiment = false },
        new ModelInput { SentimentText = "Amazing product!", Sentiment = true },
        new ModelInput { SentimentText = "Not good.", Sentiment = false },
        new ModelInput { SentimentText = "I am very happy with this.", Sentiment = true },
        new ModelInput { SentimentText = "Disappointed.", Sentiment = false },
        new ModelInput { SentimentText = "Great purchase!", Sentiment = true },
        new ModelInput { SentimentText = "Would not buy again.", Sentiment = false },
        new ModelInput { SentimentText = "Very good quality.", Sentiment = true },
        new ModelInput { SentimentText = "Not worth it.", Sentiment = false },
        new ModelInput { SentimentText = "I am delighted.", Sentiment = true },
        new ModelInput { SentimentText = "Terrible service.", Sentiment = false },
        new ModelInput { SentimentText = "Excellent value.", Sentiment = true },
        new ModelInput { SentimentText = "Very dissatisfied.", Sentiment = false },
        new ModelInput { SentimentText = "I am very impressed.", Sentiment = true },
        new ModelInput { SentimentText = "Not happy.", Sentiment = false },
        new ModelInput { SentimentText = "Superb!", Sentiment = true },
        new ModelInput { SentimentText = "Awful.", Sentiment = false },
        new ModelInput { SentimentText = "I am thrilled.", Sentiment = true },
        new ModelInput { SentimentText = "Very poor.", Sentiment = false },
        new ModelInput { SentimentText = "Great product!", Sentiment = true },
        new ModelInput { SentimentText = "Not recommended.", Sentiment = false },
        new ModelInput { SentimentText = "Very happy with this.", Sentiment = true },
        new ModelInput { SentimentText = "Bad quality.", Sentiment = false },
        new ModelInput { SentimentText = "Excellent purchase.", Sentiment = true },
        new ModelInput { SentimentText = "Not satisfied.", Sentiment = false }
    ];
}

List<ModelInput> GetTestData()
{
    return
    [
        // Ambiguous examples
        new ModelInput { SentimentText = "It's okay, I guess.", Sentiment = true },
        new ModelInput { SentimentText = "Not bad, but not great either.", Sentiment = false },
        new ModelInput { SentimentText = "I have mixed feelings about this.", Sentiment = true },
        new ModelInput { SentimentText = "It's fine, nothing special.", Sentiment = false },
        new ModelInput { SentimentText = "Could be better, could be worse.", Sentiment = true }
    ];
}
