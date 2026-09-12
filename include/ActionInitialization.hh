#ifndef ActionInitialization_h
#define ActionInitialization_h 1

#include "G4VUserActionInitialization.hh"
#include "globals.hh"

class ActionInitialization
    : public G4VUserActionInitialization
{
public:

    ActionInitialization(G4double momentum, G4double refractiveIndex);

    ~ActionInitialization() override;

    void Build() const override;

private:

    G4double fMomentum;
    G4double fRefractiveIndex;
};

#endif