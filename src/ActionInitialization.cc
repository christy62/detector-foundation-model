#include "ActionInitialization.hh"

#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"

#include "G4SystemOfUnits.hh"



// Constructor


ActionInitialization::ActionInitialization(
    G4double momentum,
    G4double refractiveIndex)
    : G4VUserActionInitialization(),
      fMomentum(momentum),
      fRefractiveIndex(refractiveIndex)
{
}



// Destructor


ActionInitialization::~ActionInitialization()
{
}



// Build user actions

void ActionInitialization::Build() const
{
    auto* primaryGenerator =
        new PrimaryGeneratorAction();

    primaryGenerator->SetMomentum(
        fMomentum
    );

    SetUserAction(
        primaryGenerator
    );

    SetUserAction(
        new EventAction(fMomentum, fRefractiveIndex)
    );
}