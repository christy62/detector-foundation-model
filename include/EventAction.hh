#ifndef EventAction_h
#define EventAction_h 1

#include "G4UserEventAction.hh"
#include "globals.hh"

#include <vector>

class EventAction : public G4UserEventAction
{
public:
    EventAction(G4double momentum, G4double refractiveIndex);
    ~EventAction() override;

    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*) override;

    void AddHit(G4double x, G4double y, G4double energy);

private:
    struct Hit
    {
        G4double x;
        G4double y;
        G4double energy;
    };

    std::vector<Hit> fHits;

    G4double fMomentum;
    G4double fRefractiveIndex;
};

#endif