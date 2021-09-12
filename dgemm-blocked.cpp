#include <cstring>

const char* dgemm_desc = "Blocked dgemm.";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double* A, double* B, double* C) 
{
#ifdef COPY_OPTIMIZATION
	double* a_block = new double[block_size*block_size];
	double* b_block = new double[block_size*block_size];
	double* c_block = new double[block_size*block_size];
#endif
	int n_blocks = n / block_size;
	for (int block_i = 0; block_i < n_blocks; ++ block_i)
	{
		for (int block_j = 0; block_j < n_blocks; ++block_j)
		{
#ifdef COPY_OPTIMIZATION
			// copy the current block of C into c_block
			for (int j = 0; j < block_size; ++j)
			{
				for (int i = 0; i < block_size; ++i)
				{
					c_block[j*block_size + i] = C[(block_j*block_size + j)*n + block_i*block_size + i];
				}
			}
#endif
			
			for (int block_k = 0; block_k < n_blocks; ++block_k)
			{
#ifdef COPY_OPTIMIZATION
				// copy the current block of A and B into a_block and b_block
				for (int j = 0; j < block_size; ++j)
				{
					for (int i = 0; i < block_size; ++i)
					{
						a_block[j*block_size + i] = A[(block_k*block_size + j)*n + block_i*block_size + i];
						b_block[j*block_size + i] = B[(block_j*block_size + j)*n + block_k*block_size + i];
					}
				}
#endif
				
				for (int j = 0; j < block_size; ++j)
				{
					for (int k = 0; k < block_size; ++k)
					{
						for (int i = 0; i < block_size; ++i)
						{
#ifdef COPY_OPTIMIZATION
							c_block[j*block_size + i] +=
								a_block[k*block_size + i]*b_block[j*block_size + k];
#else
							C[(block_j*block_size + j)*n + block_i*block_size + i] +=
								A[(block_k*block_size + k)*n + block_i*block_size + i]*B[(block_j*block_size + j)*n + block_k*block_size + k];
#endif
						}
					}
				}
			}
			
#ifdef COPY_OPTIMIZATION
			// copy c_block back to the current block of C
			for (int j = 0; j < block_size; ++j)
			{
				for (int i = 0; i < block_size; ++i)
				{
					C[(block_j*block_size + j)*n + block_i*block_size + i] = c_block[j*block_size + i];
				}
			}
#endif
		}
	}
	
#ifdef COPY_OPTIMIZATION
	delete[] a_block;
	delete[] b_block;
	delete[] c_block;
#endif
}
