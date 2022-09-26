/**
 * ARQUITECTURA DE COMPUTADORES
 * 2º Grado en Ingenieria Informatica
 *
 * EJEMPLO: "Dispositivos CUDA"
 * >> Propiedades de un dispositivo CUDA
 *
 * Alumno: Rodrigo Pascual Arnaiz
 * Fecha: 22/09/2022
 *
 */



/*   Includes   */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/*   Definicion de variables   */
#define N 16


/*   Funciones   */



/**
* Funcion: propiedadesDispositivo
* Objetivo: Mustra las propiedades del dispositvo, esta funcion
*   es ejecutada llamada y ejecutada desde el host
*
* Param: INT id_dispositivo -> ID del dispotivo
* Return: void
*/
__host__ void propiedadesDispositivo(int id_dispositivo)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, id_dispositivo);

    // calculo del numero de cores (SP)
    int cuda_cores = 0;
    int multi_processor_count = deviceProp.multiProcessorCount;
    int major = deviceProp.major;
    int minor = deviceProp.minor;


    switch (major)
    {
    case 1:
        //TESLA
        cuda_cores = 8;
        break;
    case 2:
        //FERMI
        if (minor == 0)
            cuda_cores = 32;
        else
            cuda_cores = 48;
        break;
    case 3:
        //KEPLER
        cuda_cores = 192;
        break;
    case 5:
        //MAXWELL
        cuda_cores = 128;
        break;
    case 6:
        //PASCAL
        cuda_cores = 64;
        break;
    case 7:
        //VOLTA
        cuda_cores = 64;
        break;
    case 8:
        //AMPERE
        cuda_cores = 128;
        break;
    default:
        //DESCONOCIDA
        cuda_cores = 0;
    }

    if (cuda_cores == 0 ) 
    {
        printf("!!!!!dispositivo desconocido!!!!!\n");
    }
    // presentacion de propiedades
    printf("***************************************************\n");
    printf("DISPOSIRIVO %d: %s\n", id_dispositivo, deviceProp.name);
    printf("***************************************************\n");
    printf("> Capacidad de Computo \t\t\t: %d.%d\n", major, minor);
    printf("> N. de MultiProcesadores \t\t: %d \n", multi_processor_count);
    printf("> N. de CUDA Cores (%dx%d) \t\t: %d \n", cuda_cores, multi_processor_count, cuda_cores * multi_processor_count);
    printf("> Memoria Global (total) \t\t: %zu MiB\n", deviceProp.totalGlobalMem / (1 << 20));
    printf("> Memoria Compartida (por bloque) \t: %zu KiB\n", deviceProp.sharedMemPerBlock /
        (1 << 10));
    printf("> Memoria Constante (total) \t\t: %zu KiB\n", deviceProp.totalConstMem / (1 << 10));
    printf("***************************************************\n");
}



// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
    // Obetener el dispisivo cuda
    int numero_dispositivos;
    cudaGetDeviceCount(&numero_dispositivos);
    if ( numero_dispositivos != 0 ) 
    {
        printf("Se han encontrado <%d> dispositivos CUDA:\n", numero_dispositivos);
        for (int i = 0; i < numero_dispositivos; i++)
        {
            propiedadesDispositivo(i);
        }
    }
    else 
    {
        printf("!!!!!ERROR!!!!!\n");
        printf("Este ordenador no tiene dispositivo de ejecucion CUDA\n");
        printf("<pulsa [INTRO] para finalizar>");
        getchar();
        return 1;
    }




	// declaracion de arrays necesarios
	float* hst_a_matriz, * hst_b_matriz;
	float* dev_a_matriz, * dev_b_matriz;

	// reserva en el host las matriz hst
    hst_a_matriz = ( float * )malloc( N * sizeof( float ) );
    hst_b_matriz = ( float * )malloc( N * sizeof( float ) );


	// reserva en el device las matrices dev
	cudaMalloc( ( void** )&dev_a_matriz, N * sizeof( float ) );
    cudaMalloc( ( void** )&dev_b_matriz, N * sizeof( float ) );


	// inicializacion de datos del hst_a en el host
	srand( ( int )time( NULL ) );
	for ( int i = 0; i < N; i++ )
	{
		hst_a_matriz[ i ] = ( float )rand() / RAND_MAX;
	}

	// visualizacion de datos en el host
	printf( "ENTRADA (hst_A):\n" );
	for ( int i = 0; i < N; i++ )
	{
        if( i == N -1 ) 
        {
            printf( "%.2f\n", hst_a_matriz[ i ] );
        }
        else 
        {
            printf( "%.2f ", hst_a_matriz[ i ] );
        }
	}

    
	// copia de datos CPU a GPU
	cudaMemcpy( dev_a_matriz, hst_a_matriz, N * sizeof(float), cudaMemcpyHostToDevice);

    // copia de datos GPU a GPU
    cudaMemcpy( dev_b_matriz, dev_a_matriz, N * sizeof(float), cudaMemcpyDeviceToDevice);

    // copia de datos GPU a CPU
    cudaMemcpy( hst_b_matriz, dev_b_matriz, N * sizeof(float), cudaMemcpyDeviceToHost);

    // visualizacion de datos en el Device
    printf( "SALIDA (hst_b):\n" );
    for ( int i = 0; i < N; i++ )
    {
        if ( i == N - 1 )
        {
            printf( "%.2f\n", hst_b_matriz[i] );
        }
        else
        {
            printf( "%.2f ", hst_b_matriz[i] );
        }
    }

     // salida
	time_t fecha;
	time( &fecha );
	printf( "***************************************************\n" );
	printf( "Programa ejecutado el: %s\n", ctime( &fecha ) );
	printf( "<pulsa [INTRO] para finalizar>" );
	getchar();
	return 0;
}

