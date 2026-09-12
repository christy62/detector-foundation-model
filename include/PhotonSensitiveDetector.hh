#ifndef PhotonSensitiveDetector_h
#define PhotonSensitiveDetector_h 1

#include "G4VSensitiveDetector.hh"
#include "globals.hh"

class G4Step;
class G4TouchableHistory;

class PhotonSensitiveDetector : public G4VSensitiveDetector
{
public:
    PhotonSensitiveDetector(const G4String& name);
    ~PhotonSensitiveDetector() override;

    G4bool ProcessHits(
        G4Step* step,
        G4TouchableHistory* history) override;
};

#endif