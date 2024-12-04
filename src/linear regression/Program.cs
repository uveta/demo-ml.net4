using linear_regression.Models;
using Microsoft.ML;

try
{
    // Initialize MLContext
    var mlContext = new MLContext();

    // Load and prepare data
    var trainingData = GetData();

    var trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);

    // Define the learning pipeline
    var pipeline = mlContext.Transforms.Concatenate("Features", "Area", "NumberOfRooms", "Age", "Floor")
        .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

    // Train the model
    var model = pipeline.Fit(trainingDataView);

    // Evaluate the model
    var predictions = model.Transform(trainingDataView);
    var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Price");

    // Output evaluation metrics
    Console.WriteLine($"R^2: {metrics.RSquared}");
    Console.WriteLine($"MAE: {metrics.MeanAbsoluteError}");
    Console.WriteLine($"MSE: {metrics.MeanSquaredError}");
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
return;

List<ModelInput> GetData()
{
    return
    [
        new ModelInput
        {
            Area = 50,
            NumberOfRooms = 2,
            Age = 10,
            Floor = 1,
            Price = 200000
        },
        new ModelInput
        {
            Area = 70,
            NumberOfRooms = 3,
            Age = 5,
            Floor = 2,
            Price = 300000
        },
        new ModelInput
        {
            Area = 90,
            NumberOfRooms = 4,
            Age = 2,
            Floor = 3,
            Price = 400000
        },
        new ModelInput
        {
            Area = 120,
            NumberOfRooms = 5,
            Age = 1,
            Floor = 4,
            Price = 500000
        },
        new ModelInput
        {
            Area = 60,
            NumberOfRooms = 2,
            Age = 15,
            Floor = 1,
            Price = 180000
        },
        new ModelInput
        {
            Area = 80,
            NumberOfRooms = 3,
            Age = 8,
            Floor = 2,
            Price = 280000
        },
        new ModelInput
        {
            Area = 100,
            NumberOfRooms = 4,
            Age = 3,
            Floor = 3,
            Price = 380000
        },
        new ModelInput
        {
            Area = 110,
            NumberOfRooms = 5,
            Age = 2,
            Floor = 4,
            Price = 480000
        },
        new ModelInput
        {
            Area = 55,
            NumberOfRooms = 2,
            Age = 12,
            Floor = 1,
            Price = 190000
        },
        new ModelInput
        {
            Area = 75,
            NumberOfRooms = 3,
            Age = 6,
            Floor = 2,
            Price = 290000
        }
    ];
}
