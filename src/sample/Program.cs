using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.TorchSharp;
using TorchSharp;

namespace sample;

public static class Program
{
    // Main method
    public static void Main(string[] args)
    {
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
            var labels = context.Data.LoadFromEnumerable(
            [
                new Label { Key = "PERSON" },
                new Label { Key = "CITY" },
                new Label { Key = "COUNTRY" }
            ]);

            var dataView = context.Data.LoadFromEnumerable(
                new List<TestSingleSentenceData>(
                [
                    new TestSingleSentenceData
                    {
                        // Testing longer than 512 words.
                        Sentence = "Alice and Bob live in the USA",
                        Label = ["PERSON", "0", "PERSON", "0", "0", "0", "COUNTRY"]
                    },
                    new TestSingleSentenceData
                    {
                        Sentence = "Alice and Bob live in the USA",
                        Label = ["PERSON", "0", "PERSON", "0", "0", "0", "COUNTRY"]
                    }
                ]));
            var chain = new EstimatorChain<ITransformer>();
            var estimator = chain.Append(context.Transforms.Conversion.MapValueToKey("Label", keyData: labels))
                .Append(
                    context.MulticlassClassification.Trainers.NamedEntityRecognition(outputColumnName: "outputColumn"))
                .Append(context.Transforms.Conversion.MapKeyToValue("outputColumn"));

            var transformer = estimator.Fit(dataView);
            transformer.Dispose();

            Console.WriteLine("Success!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    private class Label
    {
        public required string Key { get; set; }
    }

    private class TestSingleSentenceData
    {
        public string Sentence;
        public string[] Label;
    }
}
