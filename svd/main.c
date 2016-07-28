static char help[] = "Solves a singular value problem with the matrix loaded from a file.\n"
  "This example works for both real and complex numbers.\n\n"
  "The command line options are:\n"
  "  -file <filename>, where <filename> = matrix file in PETSc binary form.\n\n";

#include <slepcsvd.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  SVD            svd;             /* singular value problem solver context */
  Vec            u,v;             /* left and right singular vectors */
  SVDType        type;
  PetscReal      tol,sigma,error;
  PetscInt       nsv,maxit,its,nconv,i;
  char           filename[PETSC_MAX_PATH_LEN],filepattern[PETSC_MAX_PATH_LEN];
  PetscViewer    viewer;
  PetscBool      flg,terse;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);
  SlepcInitialize(&argc,&argv,(char*)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Load the operator matrix that defines the singular value problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSingular value problem stored in file.\n\n");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-file",filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a file name with the -file option");

#if defined(PETSC_USE_COMPLEX)
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading COMPLEX matrix from a binary file...\n");CHKERRQ(ierr);
#else
  ierr = PetscPrintf(PETSC_COMM_WORLD," Reading REAL matrix from a binary file...\n");CHKERRQ(ierr);
#endif
  //ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  //ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,52749,6000,NULL,&A);CHKERRQ(ierr);
  //MatView(A,viewer);
  ierr = MatSetType(A,"mpidense");CHKERRQ(ierr);
  //ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,52749,300);CHKERRQ(ierr);
  //ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  //ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  //ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&v,&u);CHKERRQ(ierr);
  // MatGetType(A,&mattype);
  // ierr = PetscPrintf(PETSC_COMM_WORLD," mattype %s",mattype);CHKERRQ(ierr);

  // MatGetRow(A,3,&ncols,&cols,&vals);
  // for(int i =0 ; i< 3;i++)
  //   {
  //       printf("firts row of A entry %d: %.20f\n", i,vals[i]);
  //   }


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the singular value solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create singular value solver context
  */
  ierr = SVDCreate(PETSC_COMM_WORLD,&svd);CHKERRQ(ierr);

  /*
     Set operator
  */
  ierr = SVDSetOperator(svd,A);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = SVDSetFromOptions(svd);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the singular value system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SVDSolve(svd);CHKERRQ(ierr);
  ierr = SVDGetIterationNumber(svd,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = SVDGetType(svd,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = SVDGetDimensions(svd,&nsv,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested singular values: %D\n",nsv);CHKERRQ(ierr);
  ierr = SVDGetTolerances(svd,&tol,&maxit);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* show detailed info unless -terse option is given by user */
  ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
  if (terse) {
    ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
  } else {
    
   /*
     Get number of converged singular triplets
  */
  ierr = SVDGetConverged(svd,&nconv);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate singular triplets: %D\n\n",nconv);CHKERRQ(ierr);

  if (nconv>0) {
    for (i=0;i<nconv;i++) {
      /*
         Get converged singular triplets: i-th singular value is stored in sigma
      */
      ierr = SVDGetSingularTriplet(svd,i,&sigma,u,v);CHKERRQ(ierr);
      sprintf(filepattern,"U%4d.petsc", i);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filepattern,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      VecView(u,viewer);
      sprintf(filepattern,"V%4d.petsc", i);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filepattern,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
      VecView(v,viewer);
      ierr = SVDComputeError(svd,i,SVD_ERROR_RELATIVE,&error);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%4d : sigma: %12.16f      error: %12g\n",i,sigma,error);CHKERRQ(ierr);
    }
  }
  }
  ierr = SVDDestroy(&svd);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return ierr;
}

