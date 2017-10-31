#include <stdio.h>
#include <stdlib.h>

//#include "pgm.h"
#include "utils.h"


int main(int argc, char *argv[])
{
	//train
	for (int i = 1; i <= 12; i++)
		test_VLBC_SMC_PrintAttack(i, 1, 4);
	
	//-----------------

	//test
	for (int i = 1; i <= 12; i++)
		test_VLBC_SMC_PrintAttack(i, 2, 4);
	
	//-----------------

	return 0;
}