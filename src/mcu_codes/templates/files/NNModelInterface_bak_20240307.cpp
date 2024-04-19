////tensorflow libraries include
#include "tensorflow/lite/micro/kernels/micro_ops.h"
//Rick modify:24/01/25
// #include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

////include model
#include "uec_alarm_model.h"
#include "sharedData.h"
#include "uec_alarm_model.c"

////global variables declaration
tflite::ErrorReporter* error_reporter = nullptr;
tflite::MicroOpResolver* resolver = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count;
uint32_t inference_time;
int32_t _input_number = 0;
constexpr int kTensorArenaSize = g_arena_size + 10000; //+10000;
uint8_t tensor_arena[kTensorArenaSize];
// NeuralNetworkFeatureProvider* featureProvider = nullptr;
// NeuralNetworkScores* scores = nullptr;

////functions declaration
void soundsetup();
int8_t * analysissound(uint32_t sound_number, int8_t* sounds, uint32_t sound_length);
extern uint32_t ticks;
 void soundsetup(){
    //initial error_report and 
    {//
        static tflite::MicroErrorReporter micro_error_reporter;
        error_reporter = &micro_error_reporter;
        // featureProvider = new NeuralNetworkFeatureProvider();
        model = tflite::GetModel(uec_alarm_model); // load model

        //check model version
        if (model->version() != TFLITE_SCHEMA_VERSION) {
            error_reporter->Report(
                "Model provided is schema version %d not equal "
                "to supported version %d.",
                model->version(), TFLITE_SCHEMA_VERSION);
            return;
        }
    }

    //Rick modify: add the following 24/01/25
		
     {
			 static tflite::MicroMutableOpResolver<9> static_resolver;
        static_resolver.AddQuantize();
        static_resolver.AddConv2D();    
        static_resolver.AddDepthwiseConv2D();
        static_resolver.AddFullyConnected();
        static_resolver.AddDequantize();
        static_resolver.AddReshape();
        // static_resolver.AddSoftmax();
        static_resolver.AddAveragePool2D();
        static_resolver.AddMaxPool2D();
        static_resolver.AddTranspose();
        resolver = &static_resolver;
        printf("Network build done and Resolver done\n");
     }
		/*
    {
        static tflite::MicroMutableOpResolver<10> static_resolver;
        
				
        static_resolver.AddConv2D(tflite::Register_CONV_2D_INT8());    
        static_resolver.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D_INT8());
        static_resolver.AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8());
			static_resolver.AddQuantize();//new add, and need to test where to put it
        static_resolver.AddReshape();
        static_resolver.AddSoftmax(tflite::Register_SOFTMAX_INT8());
			static_resolver.AddDequantize(); 
        static_resolver.AddAveragePool2D(tflite::Register_AVERAGE_POOL_2D_INT8());
        static_resolver.AddMaxPool2D(tflite::Register_MAX_POOL_2D_INT8());
        static_resolver.AddTranspose();
        resolver = &static_resolver;
        printf("Network build done and Resolver done\n");
    }
		*/
    {//initializing model interpreter
        
        //Rick modify: no need error_reporter
        // static tflite::MicroInterpreter static_interpreter(
        // model, *resolver, tensor_arena, kTensorArenaSize, error_reporter);
        static tflite::MicroInterpreter static_interpreter(
        model, *resolver, tensor_arena, kTensorArenaSize);

        interpreter = &static_interpreter;
        printf("Interpreter done\n");    

        // Allocate memory from the tensor_arena for the model's tensors.
        TfLiteStatus allocate_status = interpreter->AllocateTensors();
        if (allocate_status != kTfLiteOk) {
            TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
            return;
        }
        printf("Allocated tensors\n");
        //please write down the following output to get real size of used arena_size
        printf("Arena used %u bytes\n", interpreter->arena_used_bytes());
    }

    {//get information(type and dimension) about model input and output
        
        // Obtain pointers to the model's input and output tensors.
        // associate input and output with interpreter's input and output
        input = interpreter->input(0);
        output = interpreter->output(0);
        printf("Dimensions\n");
        // We can view the input dimensions of the model
        printf("input->dims->size:    %u\n", input->dims->size);
        for (int i = 0; i < input->dims->size; i++) {
            printf("input->dims->data[%d]: %u\n", i, input->dims->data[i]);
        }

        // We can observe the expected data type of the input layer
        if (input->type == kTfLiteInt8) printf("int8_t input\n");
        else if (input->type == kTfLiteInt16) printf("int16_t input\n");
        else if (input->type == kTfLiteUInt8) printf("uint8_t input\n");
        else if (input->type == kTfLiteFloat16) printf("float16 input\n");
        else if (input->type == kTfLiteFloat32) printf("float32 input\n");    
        else printf("Unknown input type\n");

        printf("output->dims->size:    %u\n", output->dims->size);
        for (int i = 0; i < output->dims->size; i++) {
            printf("output->dims->data[%d]: %u\n", i, output->dims->data[i]);
        }

        // We can observe the expected data type of the input layer
        if (output->type == kTfLiteInt8) printf("int8_t output\n");
        else if (output->type == kTfLiteInt16) printf("int16_t output\n");
        else if (output->type == kTfLiteUInt8) printf("uint8_t output\n");
        else if (output->type == kTfLiteFloat16) printf("float16 output\n");
        else if (output->type == kTfLiteFloat32) printf("float32 output\n");    
        else printf("Unknown output type\n");
				// mark out by Tony Wu
        //TfLiteTensor* output = get_output(); 
        // Keep track of how many inferences we have performed.
        inference_count = 0;
    }

    return; //give cpu to caller

}

int8_t * analysissound(uint32_t input_number, int8_t* sound, uint32_t sound_length)  {
    /*
    input_number:the number of input for associating input and output
    sounds: the sound signal
    sound_length: length of sound signal
    */
    // Run the model on the sound input and make sure it succeeds.
	// modified by Tony
			RetResult res;
			res.inputNumber = 0;
      res.max_idx = 0;
      res.max_value = -128;
		
     _input_number = input_number;
     //set sound to input->data
   	for (uint32_t i = 0; i < sound_length; i++) 
    {
        input->data.int8[i] = sound[i];
    }
		printf("copy data, ticks = %d\n", ticks);
		
    //call  interpreter to do model inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    //first check whether model inference failed
    if (invoke_status != kTfLiteOk) 
    {
        error_reporter->Report("Invoke failed");
        
        // MicroPrintf("Node %s (number %d) failed to invoke with status %d",
        //               OpNameFromRegistration(registration), i, invoke_status);
      
    
        return nullptr;
    }
		printf("Invoke, ticks = %d\n", ticks);
		
    //to do model inference and get output
    if(invoke_status == kTfLiteOk)
    {
        //set local variables
        size_t max_i = 0;
				int8_t min_v = 127;
        int8_t max_v = -128;
        size_t output_dimensions = 1;
			
        tflite:TfLiteTensor* output = interpreter->output(0);
			  //printf("dim size=%d\n", output->dims->size);
        //calculating output dimensions
         for (int i = 0; i < output->dims->size; i++) 
        {
            // printf("output->dims->data[%d]: %u\n", i, output->dims->data[i]);
            output_dimensions *= output->dims->data[i];
        }
        //calculate the result
				/*
				//printf("output dimensions = %d\n", output_dimensions);
        for (size_t i = 0; i < output_dimensions; i++) 
        {
					//printf("dim %d, value = %d\n", i, output->data.int8[i]);
            if (output->data.int8[i] > max_v) {
                max_v = output->data.int8[i];
                max_i = i;
            }
            if(output->data.int8[i] < min_v){
                min_v = output->data.int8[i];
            }
        }
        */
        
    }
		
    return output->data.int8;
}