#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

#define SRAND_VALUE 1985

int global_grid_size, local_grid_size;

void set_parameters(int argc, char *argv[], int *n_generations, int process_id)
{
    if (argc != 3)
    {
        if (!process_id)
        {
            printf("Invalid number of arguments\n");
            printf("Usage: ./gol_mpi grid_size n_generations\n");
        }
        exit(EXIT_FAILURE);
    }

    global_grid_size = atoi(argv[1]);
    *n_generations = atoi(argv[2]);

    if (global_grid_size < 1 || *n_generations < 0)
    {
        if (!process_id)
        {
            printf("Invalid number of arguments\n");
            printf("Usage: ./gol_mpi grid_size n_generations\n");
        }
        exit(EXIT_FAILURE);
    }
}

int allocate_grid(char ***grid)
{
    size_t i;

    if (!(*grid = malloc(local_grid_size * sizeof(char*))))
        return 1;

    for (i = 0; i < local_grid_size; ++i)
        if (!((*grid)[i] = malloc(global_grid_size * sizeof(char))))
        {
            while (i--)
                free((*grid)[i]);
            free(*grid);
            return 1;
        }

    return 0;
}

void deallocate_grid(char **grid)
{
    size_t i;

    for (i = 0; i < local_grid_size; ++i)
        free(grid[i]);
    free(grid);
}

void make_allocations(char ***grid, char ***new_grid)
{
    if (allocate_grid(grid))
    {
        printf("Memory allocation error\n");
        exit(EXIT_FAILURE);
    }
    if (allocate_grid(new_grid))
    {
        printf("Memory allocation error\n");
        deallocate_grid(*grid);
        exit(EXIT_FAILURE);
    }
}

char get_neighbors(char **grid, size_t i, size_t j)
{
    size_t u = i - 1,
           d = i + 1,
           l = (j) ? j - 1 : global_grid_size - 1,
           r = (j != global_grid_size - 1) ? j + 1 : 0;

    return grid[u][l] + grid[u][j] + grid[u][r] +
           grid[i][l]              + grid[i][r] +
           grid[d][l] + grid[d][j] + grid[d][r] ;
}

char get_new_state(char **grid, size_t i, size_t j)
{
    char neighbors = get_neighbors(grid, i, j);

    if (grid[i][j])
    {
        if (neighbors < 2 || neighbors > 3)
            return 0;
    }
    else
    {
        if (neighbors == 3)
            return 1;
    }

    return grid[i][j];
}

int main(int argc, char *argv[])
{
    char **grid, **new_grid, **aux_ptr;
    size_t i, j;
    int k, n_generations, n_process, process_id, global_line,
        previous_process_id, next_process_id, seconds, microseconds;
    unsigned long local_sum = 0, total_sum;
    struct timeval start_timer, end_timer;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &n_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    set_parameters(argc, argv, &n_generations, process_id);

    if (process_id < global_grid_size % n_process)
    {
        local_grid_size = global_grid_size / n_process + 1;
        global_line = local_grid_size * process_id;
    }
    else
    {
        local_grid_size = global_grid_size / n_process;
        global_line = local_grid_size * process_id +
                      global_grid_size % n_process;
    }
    local_grid_size += 2;

    make_allocations(&grid, &new_grid);

    srand(SRAND_VALUE);
    for (i = 0; i < global_line * global_grid_size; ++i)
        rand();
    for (i = 1; i < local_grid_size - 1; ++i)
        for (j = 0; j < global_grid_size; ++j)
            grid[i][j] = rand() % 2;

    previous_process_id = ((process_id + n_process - 1) % n_process);
    next_process_id = ((process_id + 1) % n_process);

    gettimeofday(&start_timer, NULL);

    for (k = 0; k < n_generations; ++k)
    {
        MPI_Sendrecv(
        grid[local_grid_size - 2], global_grid_size,
        MPI_CHAR, next_process_id, 0,
        grid[0], global_grid_size,
        MPI_CHAR, previous_process_id, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(
        grid[1], global_grid_size,
        MPI_CHAR, previous_process_id, 0,
        grid[local_grid_size - 1], global_grid_size,
        MPI_CHAR, next_process_id, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (i = 1; i < local_grid_size - 1; ++i)
            for (j = 0; j < global_grid_size; ++j)
                new_grid[i][j] = get_new_state(grid, i, j);

        aux_ptr = grid;
        grid = new_grid;
        new_grid = aux_ptr;
    }

    for (i = 1; i < local_grid_size - 1; ++i)
        for (j = 0; j < global_grid_size; ++j)
            local_sum += grid[i][j];

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_UNSIGNED_LONG,
               MPI_SUM, 0, MPI_COMM_WORLD);

    gettimeofday(&end_timer, NULL);

    if (!process_id)
    {
        printf("Living cells in the end:   %lu\n", total_sum);

        seconds = (int)(end_timer.tv_sec - start_timer.tv_sec);
        microseconds = (int)(end_timer.tv_usec - start_timer.tv_usec);
        if (microseconds < 0)
        {
            seconds--;
            microseconds += 1000000;
        }
        printf("Main loops execution time: %dm%d,%06ds\n", seconds / 60,
                                                           seconds % 60,
                                                           microseconds);
    }

    deallocate_grid(grid);
    deallocate_grid(new_grid);

    MPI_Finalize();

    return 0;
}