#include "PrimaryGeneratorAction.hh"

#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"

#include "G4SystemOfUnits.hh"



// Constructor


PrimaryGeneratorAction::PrimaryGeneratorAction()
    : G4VUserPrimaryGeneratorAction(),
      fParticleGun(nullptr)
{
    
    G4int numberOfParticles = 1;

    fParticleGun =
        new G4ParticleGun(
            numberOfParticles
        );




    G4ParticleTable* particleTable =
        G4ParticleTable::GetParticleTable();

    G4ParticleDefinition* particle =
        particleTable->FindParticle("pi+");

    fParticleGun->SetParticleDefinition(
        particle
    );




    fParticleGun->SetParticleMomentum(
        5.0 * GeV
    );





    fParticleGun->SetParticlePosition(
        G4ThreeVector(
            0.0,
            0.0,
            -100.0 * mm
        )
    );



    fParticleGun->SetParticleMomentumDirection(
        G4ThreeVector(
            0.0,
            0.0,
            1.0
        )
    );
}



PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fParticleGun;
}



void PrimaryGeneratorAction::GeneratePrimaries(
    G4Event* event
)
{
    fParticleGun->GeneratePrimaryVertex(
        event
    );
}

void PrimaryGeneratorAction::SetMomentum(G4double momentum)
{
    fParticleGun->SetParticleMomentum(momentum);
}