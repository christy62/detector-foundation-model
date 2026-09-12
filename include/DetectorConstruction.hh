#ifndef DetectorConstruction_h
#define DetectorConstruction_h 1

#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

class G4VPhysicalVolume;
class G4LogicalVolume;

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:

    DetectorConstruction();
    ~DetectorConstruction() override;

    G4VPhysicalVolume* Construct() override;
    void ConstructSDandField() override;

    void SetRefractiveIndex(G4double n);

private:

    G4LogicalVolume* fAerogelLogical;
    G4LogicalVolume* fDetectorLogical;

    G4double fRefractiveIndex;

    G4double fAerogelThickness;
    G4double fAerogelSize;
    G4double fDetectorSize;
    G4double fDetectorDistance;
};

#endif