#include <cstring>

const char* dgemm_desc = "Blocked dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
	double* a_block = new double[block_size*block_size];
	double* b_block = new double[block_size*block_size];
	double* c_block = new double[block_size*block_size];
	int n_blocks = n / block_size;
	for (int block_i = 0; block_i < n_blocks; ++ block_i)
	{
		for (int block_j = 0; block_j < n_blocks; ++block_j)
		{
			// copy the current block of C into c_block
			for (int c_j = 0; c_j < block_size; ++c_j)
			{
				for (int c_i = 0; c_i < block_size; ++c_i)
				{
					c_block[c_j*block_size + c_i] = C[(block_j*block_size + c_j)*n + block_i*block_size + c_i];
				}
			}
			
			for (int block_k = 0; block_k < n_blocks; ++block_k)
			{
				// copy the current block of A and B into a_block and b_block
				for (int ab_j = 0; ab_j < block_size; ++ab_j)
				{
					for (int ab_i = 0; ab_i < block_size; ++ab_i)
					{
						a_block[ab_j*block_size + ab_i] = A[(block_k*block_size + ab_j)*n + block_i*block_size + ab_i];
						b_block[ab_j*block_size + ab_i] = B[(block_j*block_size + ab_j)*n + block_k*block_size + ab_i];
					}
				}
				
				for (int j = 0; j < block_size; ++j)
				{
					for (int k = 0; k < block_size; ++k)
					{
						for (int i = 0; i < block_size; ++i)
						{
							c_block[j*block_size + i] +=
								a_block[k*block_size + i]*b_block[j*block_size + k];
						}
					}
				}
			}
			
			// copy c_block back to the current block of C
			for (int c_j = 0; c_j < block_size; ++c_j)
			{
				for (int c_i = 0; c_i < block_size; ++c_i)
				{
					C[(block_j*block_size + c_j)*n + block_i*block_size + c_i] = c_block[c_j*block_size + c_i];
				}
			}
		}
	}
	
	delete[] a_block;
	delete[] b_block;
	delete[] c_block;
}
