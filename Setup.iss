#define AppName "DeepTools"
#define AppVersion "1.0"
#define AppPublisher "Sholukh Egor"

#define CUDA_Path "{sd}\cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive\bin"

[Setup]
AppId={{7b281e88-150e-46f6-a11d-c46d70783e4e}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputBaseFilename=install
Compression=lzma
SolidCompression=yes
WizardStyle=modern
OutputDir="."
SetupLogging=yes
ChangesEnvironment=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "app\*"; DestDir: "{app}\app"; Flags: recursesubdirs ignoreversion 
Source: "cfg\*"; DestDir: "{app}\cfg"; Flags: recursesubdirs ignoreversion
Source: "data\*"; DestDir: "{app}\data"; Flags: recursesubdirs ignoreversion

[Run]
Filename: "{src}\dist\python-3.7.9-amd64.exe"; Parameters: "/silent PrependPath=1"; Flags: waituntilterminated; StatusMsg: "Installing Python3.7"
Filename: "{src}\dist\cuda_11.6.2_511.65_windows.exe"; Parameters: "-s"; Flags: waituntilterminated; StatusMsg: "Installing CUDA 11.6"
Filename: "powershell"; Parameters: "-command ""Expand-Archive -Force -Verbose '{src}\dist\cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive.zip' '{sd}\' "" "; Flags: waituntilterminated runhidden; StatusMsg: "Installing CUDNN 8.4"
Filename: "{src}\build.bat"; Parameters: """{app}"""; Flags: waituntilterminated runhidden; StatusMsg: "Installing pip packages"

Filename: "{cmd}"; Parameters: "/k cd ""{app}"" && env\Scripts\activate.bat"; Description: "Run environment"; Flags: postinstall

[UninstallRun]
Filename: "{src}\dist\python-3.7.9-amd64.exe"; RunOnceId: "RemovePython";  Parameters: "/uninstall /silent"; Flags: waituntilterminated; StatusMsg: "Uninstall Python3.7"


[UninstallDelete]
Type: filesandordirs; Name: "{app}\env"
Type: filesandordirs; Name: "{sd}\cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive"

[Registry]
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{#CUDA_Path}"; Check: NeedsAddPath(ExpandConstant('{#CUDA_Path}'))

[Code]

procedure CurStepChanged(CurStep: TSetupStep);
var
  logfilepathname, logfilename, newfilepathname: string;
begin
  logfilepathname := ExpandConstant('{log}');
  logfilename := ExtractFileName(logfilepathname);
  newfilepathname := ExpandConstant('{src}\') + logfilename;

  if CurStep = ssDone then
  begin
    FileCopy(logfilepathname, newfilepathname, false);
  end;
end;

function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', OrigPath)
  then begin
    Result := True;
    exit;
  end;
  { look for the path with leading and trailing semicolon }
  { Pos() returns 0 if not found }
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;