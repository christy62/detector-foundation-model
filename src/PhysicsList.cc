#include "PhysicsList.hh"

#include "G4DecayPhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4OpticalPhysics.hh"

#include "G4SystemOfUnits.hh"


// Constructor

PhysicsList::PhysicsList()
    : G4VModularPhysicsList()
{
    

    RegisterPhysics(
        new G4EmStandardPhysics()
    );


    

    RegisterPhysics(
        new G4DecayPhysics()
    );


    

    G4OpticalPhysics* opticalPhysics =
        new G4OpticalPhysics();

    RegisterPhysics(
        opticalPhysics
    );
}


PhysicsList::~PhysicsList()
{
}