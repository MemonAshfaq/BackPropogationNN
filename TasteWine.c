#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

//#define __DEBUG__
#ifdef __DEBUG__
    #define    DEBUG_PRINTF(f,...)    printf(f,##__VA_ARGS__)
#else
    #define    DEBUG_PRINTF(f,...)
#endif

#define FILENAME "random_wine.csv"

#define INPUTS 	13
#define OUTPUTS	3
#define HIDDEN	2
#define ROWS	178

#define L_RATE 	0.2
#define BIAS 	1
#define N_EPOCH	50

struct data {
    double input[13];
    int teach;
};

struct minmax {
    double min; 
    double max;
};

double iw[HIDDEN][INPUTS+BIAS];//input weights + 1 bias
double normalized_ip[ROWS][INPUTS];//inputs normalized in range [0,1]

double hidden_op[HIDDEN];// hidden layer output
double hidden_delta[HIDDEN];// hidden layer output

double ow[OUTPUTS][HIDDEN+BIAS];//output weights
int expected_op[OUTPUTS];//expected outputs
double generated_op[OUTPUTS];//generated outputs
double op_delta[OUTPUTS];// delta for output layer

// sigmoid serves as activation function
double sigmoid(double x) 
{
    return (1.0 / (1.0 + exp(-x)));
}

double transfer_derivative(double output)
{
	return (output * (1.0 - output));
}

void init_weights()
{
	int i,j;
	DEBUG_PRINTF("initialize input weights.\n");
    for (i = 0;i<HIDDEN; i++) {
        for (j = 0; j<INPUTS+BIAS; j++) {
            iw[i][j] = ((double)rand() / 32767.0);
			if(j==INPUTS+BIAS-1)
				DEBUG_PRINTF("BIAS ");
            DEBUG_PRINTF("IW %f\n", iw[i][j]);
        }
        DEBUG_PRINTF("\n");		
    }
	
	DEBUG_PRINTF("initialize output weights.\n");
    for (i = 0; i<OUTPUTS; i++) {
        for (j = 0; j<HIDDEN+BIAS; j++) {
            ow[i][j] = ((double)rand() / 32767.0);
			if(j==HIDDEN+BIAS-1)
				DEBUG_PRINTF("BIAS ");
            DEBUG_PRINTF("OW %f\n", ow[i][j]);
        }
		DEBUG_PRINTF("\n");
    }
}

void read_csv(char *filename, struct data* ptr) {
    FILE *file;
    file = fopen(filename, "r");

    char line[4098];
    while (fgets(line, 4098, file))
    {
        int j = 0;
        const char* tok;
        for (tok = strtok(line, ","); (tok && *tok); j++, tok = strtok(NULL, ","))
        {
            if (j < INPUTS)
                ptr->input[j] = atof(tok);
            else
                ptr->teach = atoi(tok);
        }
        ptr++;
    }
}

void print_training_data(struct data train_data[])
{
	int i,j;
    for (i = 0; i<ROWS; i++)
    {
        for (j = 0; j<INPUTS; j++) 
			DEBUG_PRINTF("%f ", train_data[i].input[j]);
        DEBUG_PRINTF("%d\n", train_data[i].teach);
    }
}

void dataset_minmax(struct minmax* ptr, struct data train_data[],int column)
{
	int i;
	ptr[column].min = train_data[0].input[column];
	ptr[column].max = train_data[0].input[column];
	
 	for(i=0;i<ROWS;i++)
	{
		if (ptr[column].max < train_data[i].input[column])
			ptr[column].max = train_data[i].input[column];
		if (ptr[column].min > train_data[i].input[column])
			ptr[column].min = train_data[i].input[column];
	}
	DEBUG_PRINTF("%.1f %.1f\n",ptr[column].min, ptr[column].max);
}

void normalize_dataset(struct data train_data[],struct minmax mm[])
{
	int i,j;
	for (i = 0; i<ROWS; i++) {
		for(j=0;j<INPUTS;j++){
		normalized_ip[i][j] = (train_data[i].input[j] - mm[j].min) / (mm[j].max - mm[j].min);
		DEBUG_PRINTF("%f ",normalized_ip[i][j]);
		}
		DEBUG_PRINTF("\n");
	}
	DEBUG_PRINTF("\n");
}

void generate_expected(int output)
{
	int i;
	DEBUG_PRINTF("expected: %d ",output);
	for (i = 0; i<OUTPUTS; i++){
		expected_op[i] = 0;
		if( i == (output-1) )
			expected_op[i] = 1;
		DEBUG_PRINTF("%d ",expected_op[i]);
		}
	DEBUG_PRINTF("\n");
}

void forward_propogate(int row)
{
	double sum;
	int i,j;
	
	for (i = 0; i<HIDDEN; i++) {
		sum = 0.0;
		for (j = 0; j<INPUTS; j++)
			sum += iw[i][j] * normalized_ip[row][j];//ni[raw][j]
		sum += iw[i][j];//add bias
		//DEBUG_PRINTF("sum i/p: %f\n",sum);
		hidden_op[i] = sigmoid(sum);// sigmoid serves as the activation function
		DEBUG_PRINTF("hidden o/p: %f\n",hidden_op[i]);
	}
	
	for (i = 0; i<OUTPUTS; i++) {
		sum = 0.0;
		for (j = 0; j<HIDDEN; j++)
			sum += ow[i][j] * hidden_op[j];
		sum += ow[i][j];//add bias
		//DEBUG_PRINTF("sum hidden: %f\n",sum);
		generated_op[i] = sigmoid(sum);
		DEBUG_PRINTF("final o/p: %f ",generated_op[i]);
	}
	DEBUG_PRINTF("\n");
}

void error_propogate()
{
	int i,j;
	double errtemp;
	for (i = 0; i<OUTPUTS; i++) {
		errtemp = expected_op[i] - generated_op[i];
		op_delta[i] = errtemp * transfer_derivative(generated_op[i]);
		DEBUG_PRINTF("op_delta : %f\n",op_delta[i]);
	}
	for (i = 0; i<HIDDEN; i++) {
		errtemp = 0.0;
		for (j = 0; j<OUTPUTS; j++)
			errtemp += op_delta[j] * ow[j][i];
		hidden_delta[i] = errtemp * transfer_derivative(hidden_op[i]);
		DEBUG_PRINTF("hidden_delta : %f\n",hidden_delta[i]);
	}
}

void update_weights(int row)
{
	int i,j;
	DEBUG_PRINTF("UPDATE input WEIGHTS\n");

	for (i = 0; i<HIDDEN; i++) {
		for (j = 0; j<INPUTS; j++) {
			DEBUG_PRINTF("before update: %f ",iw[i][j]);			
			iw[i][j] = iw[i][j] + L_RATE * hidden_delta[i] * normalized_ip[row][j];
			DEBUG_PRINTF("after update: %f\n",iw[i][j]);
		}
		iw[i][j] = iw[i][j] + L_RATE * hidden_delta[i] ;
	}

	DEBUG_PRINTF("UPDATE output WEIGHTS\n");
	
	for (i = 0; i<OUTPUTS; i++) {
		for (j = 0; j<HIDDEN; j++) {
			DEBUG_PRINTF("before update: %f ",ow[i][j]);
			ow[i][j] = ow[i][j] + L_RATE * op_delta[i] * hidden_op[j];
			DEBUG_PRINTF("after update: %f\n",ow[i][j]);			
		}
		ow[i][j] = ow[i][j] + L_RATE * op_delta[i] ;		
	}

}

void print_weights()
{
	int i,j;
	DEBUG_PRINTF("printing input weights.\n");
    for (i = 0;i<HIDDEN; i++) {
        for (j = 0; j<INPUTS+BIAS; j++) {
            if(j==INPUTS+BIAS-1)
				DEBUG_PRINTF("BIAS ");
            DEBUG_PRINTF("IW %f\n", iw[i][j]);
        }
        DEBUG_PRINTF("\n");		
    }
	
	DEBUG_PRINTF("printing output weights.\n");
    for (i = 0; i<OUTPUTS; i++) {
        for (j = 0; j<HIDDEN+BIAS; j++) {
			if(j==HIDDEN+BIAS-1)
				DEBUG_PRINTF("BIAS ");
            DEBUG_PRINTF("OW %f\n", ow[i][j]);
        }
		DEBUG_PRINTF("\n");
    }
}

int predict()
{
	int i,index=0;
	double max = generated_op[0];
	for(i=0;i<OUTPUTS;i++)
	{
		if (max < generated_op[i])
		{
			max = generated_op[i];
			index = i;
		}
	}
	return (index+1);
}

int main()
{
	char fname[32] = FILENAME;
    struct data train_data[ROWS];
	struct minmax min_max[INPUTS];
	int i,j,row,column,training=0;
	float accurate=0;
	double min, max;
	srand(time(0));
	read_csv(fname,train_data);

	for(column=0;column<INPUTS;column++)
		dataset_minmax(min_max,train_data,column);
	normalize_dataset(train_data,min_max);
	init_weights();
	
	print_weights();

	while(training++ < N_EPOCH)
	{
		for(row=0;row<ROWS;row++)
		{
			forward_propogate(row);
			generate_expected(train_data[row].teach);
			error_propogate();
			update_weights(row);
		}
	}
	print_weights();
	
	strcpy(fname,"wine_data.csv");
	read_csv(fname,train_data);
	for(column=0;column<INPUTS;column++)
		dataset_minmax(min_max,train_data,column);
	normalize_dataset(train_data,min_max);
	
	for(row=0;row<ROWS;row++)
	{
		forward_propogate(row);
		printf("%f ",generated_op[0]);
		printf("%f ",generated_op[1]);
		printf("%f ",generated_op[2]);
		printf("%d %d \n",predict(),train_data[row].teach);
		if( predict() == train_data[row].teach )
			accurate++;
	}
	printf("accuracy: %f ", (accurate/ROWS)*100 );
	return 0;
}