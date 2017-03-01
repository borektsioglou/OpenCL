#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;


void reduce(                                          
   local  float*    local_sums,                          
   global float*    partial_sums,
   int iter);

kernel void accelerate_flow(global float* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel, float w1, float w2)
{
  /* compute weighting factors */
  
  int grid_size = nx*ny;
  /* modify the 2nd row of the grid */
  int ii = ny - 2;

  /* get column index */
  int jj = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii * nx + jj]
      && (cells[3*grid_size + ii * nx + jj] - w1) > 0.0
      && (cells[6*grid_size + ii * nx + jj] - w2) > 0.0
      && (cells[7*grid_size + ii * nx + jj] - w2) > 0.0)
  {
    /* increase 'east-side' densities */
    cells[grid_size + ii * nx + jj] += w1;
    cells[5*grid_size + ii * nx + jj] += w2;
    cells[8* grid_size + ii * nx + jj] += w2;
    /* decrease 'west-side' densities */
    cells[3*grid_size + ii * nx + jj] -= w1;
    cells[6*grid_size + ii * nx + jj] -= w2;
    cells[7*grid_size + ii * nx + jj] -= w2;
  }
}

kernel void propagate(global float* cells,
                      global float* tmp_cells,
                      global int* obstacles,
                      int nx, int ny, float omega, local float* local_u, global float* partial_sums, int iter)
{
    //float* local_u = malloc(sizeof(float)*nx*ny);
    /* loop over the cells in the grid
    ** NB the collision step is called after
    ** the propagate step and so values of interest
    ** are in the scratch-space grid */
    float *tot_u;
    float zero = 0.0f;
    tot_u = &zero;
    float acc = 0.0f;
    float temp[NSPEEDS];
    int grid_size = nx*ny;
    int local_size = get_local_size(0);
    int ii = get_global_id(0);
    //int jj = get_global_id(1);
    int local_id  = get_local_id(0); 
    int group_id = get_group_id(0);
    local_u[local_id] = 0.0f;

	int x=ii%nx;
	int y= (int)ii/nx;

	int y_n = (y + 1) %ny; 
	int y_s = (y == 0) ?  (y + ny - 1) : (y - 1); 

	int x_e = (x + 1) % nx;      
	int x_w = (x == 0) ?  (x + nx - 1) : (x - 1);
	/* determine indices of axis-direction neighbours
	** respecting periodic boundary conditions (wrap around) */
    //printf("%d\n",sizeof(tmp_cells));

	/* propagate densities to neighbouring cells, following
	** appropriate directions of travel and writing into
	** scratch space grid */
	temp[0] = cells[ii]; /* central cell, no movement */
	temp[1] = cells[1*grid_size + y * nx + x_w]; /* east */
	temp[2] = cells[2*grid_size + y_s * nx + x]; /* north */
	temp[3] = cells[3*grid_size + y * nx + x_e]; /* west */
	temp[4] = cells[4*grid_size + y_n * nx + x]; /* south */
	temp[5] = cells[5*grid_size + y_s * nx + x_w]; /* north-east */
	temp[6] = cells[6*grid_size + y_s * nx + x_e]; /* north-west */
	temp[7] = cells[7*grid_size + y_n * nx + x_e]; /* south-west */
	temp[8] = cells[8*grid_size + y_n * nx + x_w]; /* south-east */ 

	if (obstacles[ii])
	{
		/* called after propagate, so taking values from scratch space
		** mirroring, and writing into main grid */
		tmp_cells[1*grid_size + ii] = temp[3];
		tmp_cells[2*grid_size + ii] = temp[4];
		tmp_cells[3*grid_size + ii] = temp[1];
		tmp_cells[4*grid_size + ii] = temp[2];
		tmp_cells[5*grid_size + ii] = temp[7];
		tmp_cells[6*grid_size + ii] = temp[8];
		tmp_cells[7*grid_size + ii] = temp[5];
		tmp_cells[8*grid_size + ii] = temp[6];

	}
	if (!obstacles[ii])
	{
		/* compute local density total */
		float local_density = 0.0;
		//#pragma uroll
		for (int kk = 0; kk < NSPEEDS; kk++)
		{
		  local_density += temp[kk];
		}
		float inv_local_density = 1.0 / local_density;
		/* compute x velocity component */
		float u_x = (temp[1]
		              + temp[5]
		              + temp[8]
		              - (temp[3]
		                 + temp[6]
		                 + temp[7]))
		             * inv_local_density;
		/* compute y velocity component */
		float u_y = (temp[2]
		              + temp[5]
		              + temp[6]
		              - (temp[4]
		                 + temp[7]
		                 + temp[8]))
		             * inv_local_density;

		/* velocity squared */
		float u_sq = u_x * u_x + u_y * u_y;
		//velocity
		local_u[local_id] = sqrt(u_sq);
		
		/* directional velocity components */
		float u[NSPEEDS];
		u[1] =   u_x;        /* east */
		u[2] =         u_y;  /* north */
		u[3] = - u_x;        /* west */
		u[4] =       - u_y;  /* south */
		u[5] =   u_x + u_y;  /* north-east */
		u[6] = - u_x + u_y;  /* north-west */
		u[7] = - u_x - u_y;  /* south-west */
		u[8] =   u_x - u_y;  /* south-east */

		/* equilibrium densities */
		float d_equ;
		/* zero velocity density: weight w0 */
		d_equ = 0.444444444f * local_density
		           * (1.0 - u_sq * 1.5);
		tmp_cells[ii] = temp[0]
		                                          + omega
		                                          * (d_equ - temp[0]);
         
		/* axis speeds: weight w1 */
        //#pragma unroll
		for(int kk = 1; kk < 5; kk++){
		  d_equ = 0.111111111f * local_density * (1.0 + u[kk] * 3.0
		                                   + (u[kk] * u[kk]) * 4.5
		                                   - u_sq * 1.5);
		  tmp_cells[kk*grid_size + ii] = temp[kk]
		                                            + omega
		                                            * (d_equ - temp[kk]);
		                                         
		}                                 

		/* diagonal speeds: weight w2 */
       // #pragma unroll
		for(int kk = 5; kk < 9; kk++){
		  d_equ = 0.027777777f * local_density * (1.0 + u[kk] * 3.0
		                                   + (u[kk] * u[kk]) * 4.5
		                                   - u_sq * 1.5);
		  tmp_cells[kk*grid_size + ii] = temp[kk]
		                                            + omega
		                                            * (d_equ - temp[kk]);
		  
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    

    reduce(local_u, partial_sums, iter);
    
}


void reduce(                                          
   local  float*    local_sums,                          
   global float*    partial_sums,
   int iter)                        
{                                                          
   int num_wrk_items  = get_local_size(0);                 
   int local_id       = get_local_id(0);                   
   int group_id       = get_group_id(0);
   int num_groups     = get_num_groups(0);                
   
  // float sum;                                                              
   
  for(unsigned int s=num_wrk_items/2; s>0; s>>=1) {   
  	//int index = 2*s*local_id;

    if(local_id < s)     
      local_sums[local_id] += local_sums[local_id+s];  
    
    barrier(CLK_LOCAL_MEM_FENCE);           
  }                                     
    if(local_id == 0){

      partial_sums[iter*num_groups + group_id] = local_sums[0];  
      
    }
}